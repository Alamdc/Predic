import json
from datetime import timedelta
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st

# --- IMPORTACIONES LOCALES ---
from utils.modeling import (
    TARGET, enrich_features, safe_fit, random_tune, 
    rolling_backtest, compute_estacional_tables, build_future_calendar,
    get_future_es, bootstrap_pred_interval
)
from utils.db_client import fetch_data_from_api, save_predictions_to_db

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Generador de Predicciones", layout="wide")

st.title("Generador de Proyecciones Financieras")
st.markdown("### Motor de Inferencia (XGBoost)")
st.markdown("""
Este módulo permite entrenar modelos predictivos personalizados por sucursal y generar proyecciones de flujo de efectivo.
El sistema utiliza algoritmos de Gradient Boosting para identificar patrones estacionales y tendencias.
""")
st.divider()

# ==========================================
# 1. BARRA LATERAL (FILTROS Y CONFIG)
# ==========================================
with st.sidebar:
    st.header("Parámetros de Ejecución")
    
    with st.container(border=True):
        st.subheader("1. Extracción de Datos")
        st.info("Filtros para la consulta")
        
        edo = st.number_input("Estado (edo)", min_value=0, value=None, step=1, format="%d")
        adm = st.number_input("ID Sucursal (adm)", min_value=0, value=None, step=1, format="%d")
        sucursal = st.text_input("Nombre Sucursal (Opcional)")
        
        col_d1, col_d2 = st.columns(2)
        start = col_d1.date_input("Fecha Inicio", value=None)
        end = col_d2.date_input("Fecha Fin", value=None)

        if start and end and start > end:
            st.error("Error: La fecha de inicio es posterior a la fecha fin.")

    with st.container(border=True):
        st.subheader("2. Configuración del Modelo")
        n_days = st.number_input("Horizonte de Pronóstico (Días)", min_value=1, max_value=90, value=7)
        test_size_days = st.number_input("Ventana de Validación (Días)", min_value=7, max_value=180, value=14)
        n_rolling_folds = st.number_input("Iteraciones de Backtest", min_value=1, max_value=10, value=1)
    
    with st.container(border=True):
        st.subheader("3. Optimización")
        do_tune = st.checkbox("Habilitar Tuning de Hiperparámetros", value=False, help="Aumenta el tiempo de procesamiento para buscar mejor precisión.")
        n_param_samples = st.number_input("Muestras de Tuning", min_value=2, max_value=100, value=5, disabled=not do_tune)
        early_stopping = st.number_input("Rondas de Parada Temprana", min_value=0, max_value=300, value=10)

    with st.container(border=True):
        st.subheader("4. Escenarios (Stress Test)")
        escenario_e = st.slider("Ajuste Entradas (%)", -50, 50, 0, 5)
        escenario_s = st.slider("Ajuste Salidas (%)", -50, 50, 0, 5)

    st.write("")
    btn_cargar = st.button("Iniciar Carga y Procesamiento", type="primary", use_container_width=True)

# ==========================================
# 2. CARGA DE DATOS
# ==========================================
if btn_cargar:
    payload = {
        "edo": int(edo) if edo is not None else None,
        "adm": int(adm) if adm is not None else None,
        "sucursal": sucursal if sucursal else None,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "limit": 1_000_000
    }
    with st.spinner("Conectando con API de Datos..."):
        try:
            data = fetch_data_from_api(payload)
            if data:
                st.session_state["raw_rows"] = data
                st.toast(f"Carga exitosa: {len(data)} registros obtenidos.", icon="✅")
            else:
                st.warning("La consulta no devolvió resultados. Verifique los filtros.")
                st.session_state["raw_rows"] = []
        except Exception as e:
            st.error(f"Error de conexión con API: {e}")

# Validar que existan datos en sesión
rows = st.session_state.get("raw_rows")
if not rows:
    st.info("Configure los parámetros en el panel lateral y presione 'Iniciar Carga' para comenzar.")
    st.stop()

df = pd.DataFrame(rows)
if df.empty:
    st.warning("El conjunto de datos está vacío.")
    st.stop()

# Preprocesamiento básico de fecha
if "fecha" in df.columns:
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.date

# --- VISTA PREVIA ---
with st.expander("Visualizar Datos de Entrada", expanded=False):
    st.dataframe(df.head(), use_container_width=True)

# ==========================================
# 3. PROCESAMIENTO Y PREDICCIÓN
# ==========================================
st.subheader("Ejecución del Modelo")

metrics_rows = []
param_cache = {}
all_preds = []

# Agrupamos los datos por la llave única
groups = list(df.groupby(["edo", "adm", "sucursal"], dropna=False))
total_groups = len(groups)

# --- BARRA DE PROGRESO ---
progress_container = st.container(border=True)
with progress_container:
    progress_bar = st.progress(0)
    status_text = st.empty()

for idx, ((g_edo, g_adm, g_suc), g0) in enumerate(groups):
    # Actualizar estado visual
    status_text.text(f"Procesando grupo {idx+1} de {total_groups}: Sucursal {g_suc} (ADM: {g_adm})")
    progress_bar.progress((idx + 1) / total_groups)
    
    g0 = g0.sort_values("fecha").reset_index(drop=True)

    # Validar tamaño mínimo de historia
    min_required = max(60, int(test_size_days) + 20)
    if len(g0) < min_required:
        continue

    # 1. Feature Engineering
    g, FEATS = enrich_features(g0)

    # 2. Transformación Logarítmica del Target
    y_min = float(g[TARGET].min())
    shift = -y_min + 1.0 if y_min <= 0 else 0.0
    g["_y_trans"] = np.log1p(g[TARGET] + shift)

    cutoff = len(g) - int(test_size_days)
    train_df = g.iloc[:cutoff].copy()
    valid_df = g.iloc[cutoff:].copy()

    Xtr = train_df[FEATS]; ytr = train_df["_y_trans"]
    Xva = valid_df[FEATS]; yva = valid_df["_y_trans"]

    # 3. Tuning de Hiperparámetros
    if do_tune:
        if g_adm not in param_cache:
            status_text.text(f"Ejecutando optimización de hiperparámetros para ADM {g_adm}...")
            best_params = random_tune(
                Xtr, ytr, Xva, yva, 
                n_samples=int(n_param_samples), 
                early_stopping=int(early_stopping)
            )
            param_cache[g_adm] = best_params
        params = param_cache[g_adm]
    else:
        # Parámetros rápidos por defecto
        params = dict(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror"
        )

    # 4. Entrenamiento Final
    model = xgb.XGBRegressor(**params)
    safe_fit(model, Xtr, ytr, Xva=Xva, yva=yva, early_stopping=int(early_stopping))

    # 5. Métricas y Backtest
    yva_pred = model.predict(Xva)
    rmse_val = float(np.sqrt(np.mean((yva - yva_pred)**2)))
    
    # Residuales para Bootstrap
    resid_val = (np.expm1(yva) - shift) - (np.expm1(yva_pred) - shift)

    # Backtest Rolling
    bt_input = g[FEATS + ["_y_trans"]].rename(columns={"_y_trans": "target_bt"})
    bt_metrics = rolling_backtest(
        bt_input, FEATS, "target_bt", 
        n_folds=int(n_rolling_folds), 
        params=params, 
        early_stopping=int(early_stopping)
    )

    metrics_rows.append({
        "edo": g_edo, "adm": g_adm, "sucursal": g_suc,
        "rmse_val_log": rmse_val,
        **bt_metrics
    })

    # 6. Proyección Futura
    e_tab, s_tab, e_global, s_global = compute_estacional_tables(g0)
    last_date = g0['fecha'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, int(n_days)+1)]
    s_flujo = list(g0[TARGET].values.astype(float))
    
    current_preds = []
    
    for d in future_dates:
        cal = build_future_calendar(d)
        
        e_future, s_future = get_future_es(
            cal["dia_semana"], cal["mes"], e_tab, s_tab, 
            e_global, s_global, escenario_e, escenario_s
        )

        s_series = pd.Series(s_flujo)
        xrow = {
            "entradas": e_future, "salidas": s_future, **cal,
            "media_movil_3": s_series.tail(3).mean(),
            "media_movil_5": s_series.tail(5).mean(),
            "media_movil_10": s_series.tail(10).mean(),
            "media_movil_14": s_series.tail(14).mean(),
            "lag_1": s_flujo[-1], 
            "lag_2": s_flujo[-2] if len(s_flujo)>1 else s_flujo[-1],
            "lag_3": s_flujo[-3] if len(s_flujo)>2 else s_flujo[-1],
            "lag_5": s_flujo[-5] if len(s_flujo)>4 else s_flujo[-1],
        }
        
        xdf_row, _ = enrich_features(pd.DataFrame([xrow]))
        xdf_row = xdf_row[FEATS]
        
        yhat_log = float(model.predict(xdf_row)[0])
        yhat = float(np.expm1(yhat_log) - shift)
        
        yhat_, lo, hi = bootstrap_pred_interval(resid_val.values, yhat)
        
        current_preds.append({
            "edo": g_edo, "adm": g_adm, "sucursal": g_suc,
            "fecha_predicha": d,
            "prediccion": yhat_, "limite_inferior": lo, "limite_superior": hi,
            "version_modelo": f"xgb_v2_adm_{g_adm}",
            "entrenado_hasta": last_date,
            "variables_json": json.dumps({**xrow, "shift": shift})
        })
        s_flujo.append(yhat_)
    
    all_preds.append(pd.DataFrame(current_preds))

progress_bar.progress(100)
status_text.text("Procesamiento completado con éxito.")

# ==========================================
# 4. RESULTADOS Y EXPORTACIÓN
# ==========================================
st.divider()

if all_preds:
    all_preds_df = pd.concat(all_preds, ignore_index=True)
    
    # KPIs en tarjetas
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        with st.container(border=True):
            st.metric("Sucursales Procesadas", len(all_preds))
    with col_kpi2:
        with st.container(border=True):
            st.metric("Total Proyecciones", len(all_preds_df))
    with col_kpi3:
        with st.container(border=True):
            avg_rmse = np.mean([m['rmse_val_log'] for m in metrics_rows]) if metrics_rows else 0
            st.metric("RMSE Promedio (Log)", f"{avg_rmse:.4f}")

    st.subheader("Resultados Detallados")
    
    # Tabla con formato profesional
    cols_view = ["fecha_predicha", "prediccion", "limite_inferior", "limite_superior", "edo", "adm", "sucursal"]
    
    st.dataframe(
        all_preds_df[cols_view], 
        use_container_width=True,
        hide_index=True,
        column_config={
            "fecha_predicha": st.column_config.DateColumn("Fecha Proyección", format="DD/MM/YYYY"),
            "prediccion": st.column_config.NumberColumn("Flujo Proyectado", format="$ %.2f"),
            "limite_inferior": st.column_config.NumberColumn("Límite Inferior (95%)", format="$ %.2f"),
            "limite_superior": st.column_config.NumberColumn("Límite Superior (95%)", format="$ %.2f"),
            "edo": st.column_config.NumberColumn("Estado", format="%d"),
            "adm": st.column_config.NumberColumn("ID Sucursal", format="%d"),
        }
    )

    # Botones de Acción
    st.write("")
    col_dl, col_db = st.columns(2)
    
    with col_dl:
        csv = all_preds_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Reporte (CSV)", 
            data=csv, 
            file_name="predicciones_flujo.csv", 
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
    
    with col_db:
        if st.button("Archivar Proyecciones en Base de Datos", use_container_width=True):
            with st.spinner("Escribiendo en base de datos PostgreSQL..."):
                try:
                    save_predictions_to_db(all_preds_df)
                    st.success(f"Operación exitosa: {len(all_preds_df)} registros archivados.")
                except Exception as e:
                    st.error(f"Error en operación de guardado: {e}")

else:
    st.info("El proceso finalizó sin generar proyecciones. Esto puede deberse a filtros muy restrictivos o falta de datos históricos suficientes.")