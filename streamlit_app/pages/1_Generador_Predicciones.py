import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import warnings
import json
import xgboost as xgb
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

# --- IMPORTACIONES LOCALES ---
from utils.modeling import (
    TARGET, enrich_features, safe_fit, 
    compute_estacional_tables, build_future_calendar,
    get_future_es, bootstrap_pred_interval
)
from utils.db_client import fetch_data_from_api, save_predictions_to_db

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Proyecciones XGBoost", layout="wide")
warnings.filterwarnings("ignore")

# --- ESTILOS Y TÍTULO ---
st.title("Generador de Proyecciones Financieras")
st.markdown("### Motor de Inferencia (XGBoost) con Limpieza")
st.markdown("""
Este modulo utiliza algoritmos de Gradient Boosting (XGBoost) para identificar patrones complejos.
Incluye pre-procesamiento con Isolation Forest para eliminar distorsiones por flujos atipicos.
""")
st.divider()

# ==========================================
# 1. BARRA LATERAL (FILTROS Y CONFIG)
# ==========================================
with st.sidebar:
    st.header("Parametros de Ejecucion")
    
    with st.container(border=True):
        st.subheader("1. Busqueda de Sucursal")
        st.info("Puede buscar por ADM, Estado o Nombre")
        
        edo_code = st.number_input("Estado (edo)", min_value=0, value=0, step=1, format="%d")
        adm_code = st.number_input("ID Admin (adm)", min_value=0, value=0, step=1, format="%d", help="El ID unico numerico de la sucursal")
        sucursal_txt = st.text_input("Nombre Sucursal (Opcional)", value="", help="Busqueda exacta por nombre")

    with st.container(border=True):
        st.subheader("2. Configuracion del Modelo")
        days_history = st.slider("Dias de historia a usar", 60, 2000, 365, help="Cuantos dias hacia atras usar para entrenar.")
        forecast_horizon = st.number_input("Horizonte de Pronostico (Dias)", min_value=1, max_value=60, value=7)
        # Nota: XGBoost no usa "ciclo estacional" explícito como parámetro, lo aprende de los features, 
        # pero mantenemos la estética similar.

    with st.container(border=True):
        st.subheader("3. Configuracion de Limpieza")
        usar_limpieza = st.checkbox("Activar Limpieza de Datos", value=True)
        contamination = st.slider("Sensibilidad Isolation Forest", 0.01, 0.10, 0.02, disabled=not usar_limpieza)
        umbral_millones = st.number_input("Umbral Maximo (Millones)", min_value=0.5, max_value=50.0, value=2.0, step=0.5, disabled=not usar_limpieza)

    st.write("")
    btn_buscar = st.button("Buscar Datos", type="primary", use_container_width=True)

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def aplicar_limpieza_hibrida(df, col_target, contaminacion, umbral_m):
    """
    Limpia una serie temporal usando Isolation Forest + Regla de Negocio.
    """
    g = df.copy().sort_values('fecha')
    umbral_valor = umbral_m * 1_000_000
    
    # 1. Isolation Forest (Estadístico)
    model = IsolationForest(contamination=contaminacion, random_state=42)
    X = g[[col_target]].values
    preds = model.fit_predict(X) # -1 es outlier
    
    # 2. Regla Dura (Umbral de Millones)
    mask_exceso = g[col_target].abs() > umbral_valor
    preds[mask_exceso] = -1
    
    g['is_outlier'] = preds
    g['original'] = g[col_target]
    
    # 3. Imputar (Interpolación)
    g.loc[g['is_outlier'] == -1, col_target] = np.nan
    g[col_target] = g[col_target].interpolate(method='linear', limit_direction='both')
    
    return g

def plot_xgboost_results(history_df, pred_df, title_text):
    fig = go.Figure()
    # Historia (Últimos 180 días visuales)
    viz_history = history_df.tail(180)
    fig.add_trace(go.Scatter(
        x=viz_history.index, y=viz_history[TARGET],
        mode='lines', name='Historia Entrenada', line=dict(color='#6c757d', width=1.5)
    ))
    # Intervalos
    fig.add_trace(go.Scatter(
        x=pred_df['fecha_predicha'], y=pred_df['limite_superior'],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=pred_df['fecha_predicha'], y=pred_df['limite_inferior'],
        line=dict(width=0), mode='lines', fill='tonexty', fillcolor='rgba(40, 167, 69, 0.15)', # Verde para XGB
        name='Intervalo Confianza (95%)', hoverinfo='skip'
    ))
    # Predicción
    fig.add_trace(go.Scatter(
        x=pred_df['fecha_predicha'], y=pred_df['prediccion'],
        mode='lines+markers', name='Proyeccion XGBoost',
        line=dict(color='#28a745', width=2.5), marker=dict(size=6)
    ))
    fig.update_layout(
        title=title_text, xaxis_title="Fecha", yaxis_title="Flujo ($)",
        height=450, hovermode="x unified", template="plotly_white"
    )
    return fig

# ==========================================
# 2. LÓGICA PRINCIPAL
# ==========================================

if 'search_data_xgb' not in st.session_state:
    st.session_state['search_data_xgb'] = None

if btn_buscar:
    if (edo_code == 0) and (adm_code == 0) and (not sucursal_txt):
        st.error("Debe ingresar al menos un criterio (Estado, ADM o Nombre).")
    else:
        with st.spinner("Conectando con base de datos tdv_data..."):
            
            payload = {
                "edo": edo_code if edo_code != 0 else None,
                "adm": adm_code if adm_code != 0 else None,
                "sucursal": sucursal_txt if sucursal_txt else None,
                "start": None, 
                "end": None,
                "limit": 10000 
            }
            
            try:
                rows = fetch_data_from_api(payload)
                if not rows:
                    st.warning("La consulta no devolvio registros.")
                    st.session_state['search_data_xgb'] = None
                else:
                    df = pd.DataFrame(rows)
                    if "fecha" in df.columns:
                        df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
                    df = df.sort_values('fecha')
                    st.session_state['search_data_xgb'] = df
                    st.rerun()
            except Exception as e:
                st.error(f"Error de conexion: {e}")

# --- SELECCIÓN Y ENTRENAMIENTO ---
if st.session_state['search_data_xgb'] is not None:
    df_raw = st.session_state['search_data_xgb']
    
    st.success(f"Se encontraron {len(df_raw)} registros historicos.")
    
    # Selector de Sucursal
    col_sel, col_act = st.columns([3, 1])
    with col_sel:
        opciones = df_raw[['adm', 'sucursal']].drop_duplicates().sort_values('adm')
        opciones['label'] = opciones['adm'].astype(str) + " - " + opciones['sucursal']
        
        selection = st.selectbox(
            "Seleccione la Sucursal a Modelar:", 
            options=opciones['label'].tolist()
        )
        adm_seleccionado = int(selection.split(" - ")[0])
        
    # Filtrar DF para esa sucursal
    # Nota: XGBoost necesita dataframe con columnas, no serie
    df_target = df_raw[df_raw['adm'] == adm_seleccionado].copy()
    
    # Convertir fecha a datetime para operaciones
    df_target['fecha'] = pd.to_datetime(df_target['fecha'])
    df_target = df_target.sort_values('fecha').reset_index(drop=True)
    
    # Recorte de Historia
    if len(df_target) > days_history:
        df_target = df_target.tail(days_history).reset_index(drop=True)

    # --- VISUALIZACIÓN DE LIMPIEZA PREVIA ---
    if usar_limpieza:
        with st.expander("Ver Analisis de Limpieza (Anomalias)", expanded=True):
            df_preview_clean = aplicar_limpieza_hibrida(df_target, TARGET, contamination, umbral_millones)
            n_out = (df_preview_clean['is_outlier'] == -1).sum()
            
            st.markdown(f"**Anomalias detectadas:** {n_out}")
            
            fig_clean = go.Figure()
            outliers = df_preview_clean[df_preview_clean['is_outlier'] == -1]
            fig_clean.add_trace(go.Scatter(x=outliers['fecha'], y=outliers['original'], mode='markers', name='Anomalia Eliminada', marker=dict(color='red', size=8, symbol='x')))
            fig_clean.add_trace(go.Scatter(x=df_preview_clean['fecha'], y=df_preview_clean[TARGET], mode='lines', name='Flujo Limpio', line=dict(color='green')))
            fig_clean.update_layout(height=350, title="Impacto de la Limpieza", template="plotly_white")
            st.plotly_chart(fig_clean, use_container_width=True)

    with col_act:
        st.write("") 
        st.write("")
        btn_entrenar = st.button("Entrenar Modelo", type="primary")

    if btn_entrenar:
        if len(df_target) < 60:
            st.error("No hay datos suficientes (minimo 60 dias) para entrenar XGBoost.")
            st.stop()
            
        sucursal_nombre = df_target['sucursal'].iloc[0] 
        edo_val = int(df_target['edo'].iloc[0])

        with st.status(f"Procesando modelo XGBoost para {selection}...", expanded=True) as status:
            try:
                # 1. LIMPIEZA
                if usar_limpieza:
                    status.write("Aplicando Isolation Forest y Reglas de Negocio...")
                    df_train = aplicar_limpieza_hibrida(df_target, TARGET, contamination, umbral_millones)
                else:
                    df_train = df_target.copy()

                # 2. FEATURE ENGINEERING
                status.write("Generando caracteristicas temporales (Lags, Medias Moviles)...")
                g, FEATS = enrich_features(df_train)

                # 3. TRANSFORMACIÓN Y SPLIT
                y_min = float(g[TARGET].min())
                shift = -y_min + 1.0 if y_min <= 0 else 0.0
                g["_y_trans"] = np.log1p(g[TARGET] + shift)

                cutoff = len(g) - 14 # Usamos ultimos 14 dias para validar internamente
                train_df = g.iloc[:cutoff]
                valid_df = g.iloc[cutoff:]

                Xtr = train_df[FEATS]; ytr = train_df["_y_trans"]
                Xva = valid_df[FEATS]; yva = valid_df["_y_trans"]

                # 4. ENTRENAMIENTO
                status.write(f"Entrenando XGBRegressor con {len(train_df)} registros...")
                params = dict(n_estimators=500, max_depth=6, learning_rate=0.05, objective="reg:squarederror")
                model = xgb.XGBRegressor(**params)
                safe_fit(model, Xtr, ytr, Xva=Xva, yva=yva, early_stopping=20)

                # Calculo de residuales para intervalos
                yva_pred = model.predict(Xva)
                rmse_val = float(np.sqrt(np.mean((yva - yva_pred)**2)))
                resid_val = (np.expm1(yva) - shift) - (np.expm1(yva_pred) - shift)

                # 5. PROYECCIÓN FUTURA
                status.write("Calculando proyecciones futuras...")
                
                # Tablas estacionales para inputs futuros
                e_tab, s_tab, e_gl, s_gl = compute_estacional_tables(df_train)
                last_date = df_train['fecha'].max()
                future_dates = [last_date + timedelta(days=i) for i in range(1, int(forecast_horizon)+1)]
                
                s_flujo = list(df_train[TARGET].values.astype(float))
                current_preds = []

                for d in future_dates:
                    cal = build_future_calendar(d)
                    # Estimacion simple de entradas/salidas futuras (promedios estacionales)
                    e_fut, s_fut = get_future_es(cal["dia_semana"], cal["mes"], e_tab, s_tab, e_gl, s_gl, 0, 0)
                    
                    s_series = pd.Series(s_flujo)
                    n_hist = len(s_flujo)
                    
                    # Feature Construction (Manual Lags)
                    xrow = {
                        "fecha": d,
                        "entradas": e_fut, "salidas": s_fut, **cal,
                        "lag_1": s_flujo[-1] if n_hist >= 1 else 0,
                        "lag_7": s_flujo[-7] if n_hist >= 7 else s_flujo[-1],
                        "lag_14": s_flujo[-14] if n_hist >= 14 else s_flujo[-1],
                        "lag_30": s_flujo[-30] if n_hist >= 30 else s_flujo[-1],
                        "media_movil_7": s_series.tail(7).mean() if n_hist >= 7 else s_flujo[-1],
                        "media_movil_30": s_series.tail(30).mean() if n_hist >= 30 else s_flujo[-1],
                        "std_movil_7": s_series.tail(7).std() if n_hist >= 7 else 0,
                    }
                    
                    # Feature Enrichment
                    xdf_row, _ = enrich_features(pd.DataFrame([xrow]))
                    xdf_row = xdf_row.reindex(columns=FEATS, fill_value=0)
                    
                    # Prediccion
                    yhat_log = float(model.predict(xdf_row)[0])
                    yhat = float(np.expm1(yhat_log) - shift)
                    yhat_, lo, hi = bootstrap_pred_interval(resid_val.values, yhat)
                    
                    current_preds.append({
                        "fecha_predicha": d,
                        "prediccion": yhat_, "limite_inferior": lo, "limite_superior": hi
                    })
                    s_flujo.append(yhat_)

                df_pred = pd.DataFrame(current_preds)
                
                # Guardar resultados en sesión
                st.session_state['model_result_xgb'] = {
                    "df_hist": df_train.set_index('fecha'), # Usamos la limpia para graficar
                    "df_pred": df_pred,
                    "meta": {
                        "rmse": rmse_val,
                        "sucursal": sucursal_nombre,
                        "adm": adm_seleccionado,
                        "edo": edo_val,
                        "last_date": last_date
                    }
                }
                status.update(label="Modelo completado exitosamente", state="complete", expanded=False)
                
            except Exception as e:
                st.error(f"Error en modelado: {e}")
                st.stop()

# --- MOSTRAR RESULTADOS ---
if 'model_result_xgb' in st.session_state:
    res = st.session_state['model_result_xgb']
    meta = res['meta']
    df_view = res['df_pred']
    
    st.divider()
    st.subheader(f"Resultados del Pronostico: {meta['adm']} - {meta['sucursal']}")
    
    # Métricas Técnicas
    k1, k2, k3 = st.columns(3)
    k1.metric("RMSE (Log)", f"{meta['rmse']:.4f}", help="Error cuadratico medio en escala logaritmica (menor es mejor)")
    k2.metric("Horizonte", f"{len(df_view)} Dias")
    k3.metric("Modelo", "XGBoost Regressor")
    
    # Grafico
    st.plotly_chart(plot_xgboost_results(res['df_hist'], res['df_pred'], "Proyeccion de Flujo (XGBoost)"), use_container_width=True)
    
    st.write("")
    st.markdown("### Tabla de Predicciones")

    st.dataframe(
        df_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "fecha_predicha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
            "prediccion": st.column_config.NumberColumn("Prediccion Flujo", format="$ %.2f"),
            "limite_inferior": st.column_config.NumberColumn("Limite Inferior (95%)", format="$ %.2f"),
            "limite_superior": st.column_config.NumberColumn("Limite Superior (95%)", format="$ %.2f")
        }
    )
    
    st.write("")
    
    if st.button("Guardar Predicciones en BD", type="secondary", use_container_width=True):
        df_save = res['df_pred'].copy()
        df_save["edo"] = meta["edo"]
        df_save["adm"] = meta["adm"]
        df_save["sucursal"] = meta["sucursal"]
        df_save["version_modelo"] = "xgb_iso_v2"
        df_save["entrenado_hasta"] = meta["last_date"]
        df_save["variables_json"] = json.dumps({"rmse": meta["rmse"]})
        
        if save_predictions_to_db(df_save):
            st.success(f"Datos guardados correctamente para {meta['sucursal']}")
        else:
            st.error("Error al guardar en base de datos.")