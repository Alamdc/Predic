import json
from datetime import timedelta
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# --- LOCAL IMPORTS ---
from utils.modeling import (
    TARGET, enrich_features, safe_fit, random_tune, 
    rolling_backtest, compute_estacional_tables, build_future_calendar,
    get_future_es, bootstrap_pred_interval
)
from utils.db_client import fetch_data_from_api, save_predictions_to_db

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Generador de Predicciones", layout="wide")

st.title("Generador de Proyecciones Financieras")
st.markdown("### Motor de Inferencia (XGBoost) con Limpieza de Anomalias")
st.markdown("""
Este modulo permite entrenar modelos predictivos personalizados por sucursal, aplicar limpieza de anomalias (Isolation Forest + Reglas de Negocio) y generar proyecciones de flujo de efectivo.
""")
st.divider()

# ==========================================
# 1. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("Parametros de Ejecucion")
    
    with st.container(border=True):
        st.subheader("1. Extraccion de Datos")
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
        st.subheader("2. Configuracion del Modelo")
        n_days = st.number_input("Horizonte de Pronostico (Dias)", min_value=1, max_value=90, value=7)
        
        st.markdown("---")
        st.write("**Configuracion de Limpieza**")
        contamination = st.slider("Sensibilidad Isolation Forest", 0.01, 0.10, 0.02, help="Porcentaje de datos a considerar anomalos estadisticamente")
        umbral_millones = st.number_input("Umbral Maximo (Millones)", min_value=0.5, max_value=50.0, value=2.0, step=0.5, help="Cualquier flujo mayor a esta cantidad sera eliminado y suavizado.")

    st.write("")
    # Step 1 Button
    btn_cargar = st.button("1. Iniciar Carga de Datos", type="primary", use_container_width=True)

# ==========================================
# HELPER FUNCTION: ISOLATION FOREST + THRESHOLD
# ==========================================
def aplicar_isolation_forest_hibrido(df, col_target, contaminacion, umbral_m):
    """
    Detects anomalies using both Isolation Forest and a Hard Threshold.
    """
    model = IsolationForest(contamination=contaminacion, random_state=42)
    umbral_valor = umbral_m * 1_000_000 # Convertir millones a unidades
    
    # Process by group to avoid mixing scales
    grupos = df.groupby(['edo', 'adm', 'sucursal'])
    dfs_limpios = []
    
    for name, group in grupos:
        g = group.copy().sort_values('fecha')
        
        # 1. Statistical Detection (Isolation Forest)
        X = g[[col_target]].values
        preds = model.fit_predict(X) # -1 is outlier
        
        # 2. Business Rule Detection (Threshold)
        # Si el valor absoluto supera el umbral, forzamos a -1 (outlier)
        mask_exceso = g[col_target].abs() > umbral_valor
        preds[mask_exceso] = -1
        
        g['is_outlier'] = preds
        
        # 3. Imputation (Interpolation)
        g['original'] = g[col_target] # Save original for comparison
        g.loc[g['is_outlier'] == -1, col_target] = np.nan
        g[col_target] = g[col_target].interpolate(method='linear', limit_direction='both')
        
        dfs_limpios.append(g)
        
    return pd.concat(dfs_limpios)

# ==========================================
# STEP 1: LOAD DATA
# ==========================================
if btn_cargar:
    payload = {
        "edo": int(edo) if edo is not None else None,
        "adm": int(adm) if adm is not None else None,
        "sucursal": sucursal if sucursal else None,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "limit": 1000000
    }
    with st.spinner("Conectando con API de Datos..."):
        data = fetch_data_from_api(payload)
        if data:
            df = pd.DataFrame(data)
            if "fecha" in df.columns:
                df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
            
            # Save to session
            st.session_state['step1_df'] = df
            # Clear subsequent steps
            st.session_state.pop('step2_df', None)
            st.session_state.pop('step3_results', None)
            st.success(f"Carga exitosa: {len(df)} registros obtenidos.")
        else:
            st.warning("La consulta no devolvio resultados.")

# Display Data Preview if Step 1 is done
if 'step1_df' in st.session_state:
    df_step1 = st.session_state['step1_df']
    
    with st.expander("Visualizar Datos de Entrada", expanded=False):
        st.dataframe(df_step1.head(), use_container_width=True)
    
    st.divider()

    # ==========================================
    # STEP 2: CLEANING (Isolation Forest + Reglas)
    # ==========================================
    st.subheader("Paso 2: Limpieza de Anomalias")
    st.info(f"Se eliminaran valores atipicos estadisticos Y cualquier flujo mayor a ${umbral_millones} Millones.")
    
    col_btn_clean, col_dummy = st.columns([1, 3])
    with col_btn_clean:
        if st.button("2. Ejecutar Limpieza", use_container_width=True):
            with st.spinner("Aplicando reglas y analisis estadistico..."):
                # Call the new hybrid function
                df_clean = aplicar_isolation_forest_hibrido(df_step1, 'flujo_efectivo', contamination, umbral_millones)
                
                st.session_state['step2_df'] = df_clean
                st.success("Limpieza Completada")

    # Visualization of Cleaning
    if 'step2_df' in st.session_state:
        df_step2 = st.session_state['step2_df']
        
        n_outliers = (df_step2['is_outlier'] == -1).sum()
        st.metric("Registros Corregidos (Outliers)", n_outliers)
        
        # Comparative Chart
        fig = go.Figure()
        # Outliers
        outliers = df_step2[df_step2['is_outlier'] == -1]
        fig.add_trace(go.Scatter(
            x=outliers['fecha'], y=outliers['original'],
            mode='markers', name='Dato Eliminado',
            marker=dict(color='red', size=8, symbol='x')
        ))
        # Clean Line
        fig.add_trace(go.Scatter(
            x=df_step2['fecha'], y=df_step2['flujo_efectivo'],
            mode='lines', name='Flujo Operativo (Limpio)',
            line=dict(color='blue')
        ))
        fig.update_layout(title="Comparativa: Historia Real vs Historia Limpia", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ==========================================
        # STEP 3: TRAINING AND PREDICTION
        # ==========================================
        st.subheader("Paso 3: Entrenamiento y Proyeccion")
        
        col_btn_train, col_dummy2 = st.columns([1, 3])
        with col_btn_train:
            run_train = st.button("3. Entrenar Modelo", type="primary", use_container_width=True)
        
        if run_train:
            df_final = st.session_state['step2_df']
            
            metrics_rows = []
            all_preds = []
            
            groups = list(df_final.groupby(["edo", "adm", "sucursal"]))
            total_groups = len(groups)
            
            # Progress Bar logic
            progress_container = st.container(border=True)
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            for idx, ((g_edo, g_adm, g_suc), g0) in enumerate(groups):
                status_text.text(f"Procesando {idx+1}/{total_groups}: {g_suc}")
                progress_bar.progress((idx + 1) / total_groups)
                
                g0 = g0.sort_values("fecha").reset_index(drop=True)
                
                if len(g0) < 60: continue 

                # 1. Feature Engineering
                g, FEATS = enrich_features(g0)

                # 2. Log Transform
                y_min = float(g[TARGET].min())
                shift = -y_min + 1.0 if y_min <= 0 else 0.0
                g["_y_trans"] = np.log1p(g[TARGET] + shift)

                # Train/Test Split
                cutoff = len(g) - 14
                train_df = g.iloc[:cutoff]
                valid_df = g.iloc[cutoff:]

                Xtr = train_df[FEATS]; ytr = train_df["_y_trans"]
                Xva = valid_df[FEATS]; yva = valid_df["_y_trans"]

                # Default Params
                params = dict(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    objective="reg:squarederror"
                )

                model = xgb.XGBRegressor(**params)
                safe_fit(model, Xtr, ytr, Xva=Xva, yva=yva, early_stopping=20)

                # Metrics
                yva_pred = model.predict(Xva)
                rmse = float(np.sqrt(np.mean((yva - yva_pred)**2)))
                resid_val = (np.expm1(yva) - shift) - (np.expm1(yva_pred) - shift)

                metrics_rows.append({"adm": g_adm, "rmse": rmse})

                # Future Projection
                e_tab, s_tab, e_gl, s_gl = compute_estacional_tables(g0)
                last_date = g0['fecha'].max()
                future_dates = [last_date + timedelta(days=i) for i in range(1, int(n_days)+1)]
                s_flujo = list(g0[TARGET].values.astype(float))
                
                current_preds = []
                for d in future_dates:
                    cal = build_future_calendar(d)
                    e_fut, s_fut = get_future_es(
                        cal["dia_semana"], cal["mes"], e_tab, s_tab, e_gl, s_gl, 0, 0
                    )
                    
                    s_series = pd.Series(s_flujo)
                    n_hist = len(s_flujo)
                
                    xrow = {
                        "fecha": d,
                        "entradas": e_fut, "salidas": s_fut, **cal,
                        
                        # Manual Lags
                        "lag_1": s_flujo[-1] if n_hist >= 1 else 0,
                        "lag_7": s_flujo[-7] if n_hist >= 7 else s_flujo[-1],
                        "lag_14": s_flujo[-14] if n_hist >= 14 else s_flujo[-1],
                        "lag_30": s_flujo[-30] if n_hist >= 30 else s_flujo[-1],
                        
                        # Manual Rolling Means
                        "media_movil_7": s_series.tail(7).mean() if n_hist >= 7 else s_flujo[-1],
                        "media_movil_30": s_series.tail(30).mean() if n_hist >= 30 else s_flujo[-1],
                        "std_movil_7": s_series.tail(7).std() if n_hist >= 7 else 0,
                    }
                    
                    xdf_row, _ = enrich_features(pd.DataFrame([xrow]))
                    xdf_row = xdf_row.reindex(columns=FEATS, fill_value=0)
                    
                    yhat_log = float(model.predict(xdf_row)[0])
                    yhat = float(np.expm1(yhat_log) - shift)
                    yhat_, lo, hi = bootstrap_pred_interval(resid_val.values, yhat)
                    
                    current_preds.append({
                        "edo": g_edo, "adm": g_adm, "sucursal": g_suc,
                        "fecha_predicha": d,
                        "prediccion": yhat_, "limite_inferior": lo, "limite_superior": hi,
                        "version_modelo": "xgb_iso_v1"
                    })
                    s_flujo.append(yhat_)
                
                all_preds.append(pd.DataFrame(current_preds))
            
            progress_bar.progress(100)
            status_text.text("Procesamiento completado.")
            
            if all_preds:
                res_df = pd.concat(all_preds, ignore_index=True)
                st.session_state['step3_results'] = res_df
                st.success("Proyecciones Generadas")

        # Visualization of Final Results
        if 'step3_results' in st.session_state:
            res_df = st.session_state['step3_results']
            
            st.divider()
            st.subheader("Resultados Detallados")

            # KPIs in Cards
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            with col_kpi1:
                with st.container(border=True):
                    st.metric("Total Proyecciones", len(res_df))
            with col_kpi2:
                with st.container(border=True):
                    st.metric("Sucursales", res_df['adm'].nunique())
            with col_kpi3:
                with st.container(border=True):
                    st.metric("Horizonte", f"{n_days} Dias")

            # Table configuration
            st.dataframe(
                res_df[["fecha_predicha", "prediccion", "limite_inferior", "limite_superior", "adm", "sucursal"]],
                use_container_width=True,
                column_config={
                    "fecha_predicha": st.column_config.DateColumn("Fecha Proyeccion", format="DD/MM/YYYY"),
                    "prediccion": st.column_config.NumberColumn("Flujo Proyectado", format="$ %.2f"),
                    "limite_inferior": st.column_config.NumberColumn("Limite Inferior", format="$ %.2f"),
                    "limite_superior": st.column_config.NumberColumn("Limite Superior", format="$ %.2f"),
                    "adm": st.column_config.NumberColumn("ID Sucursal", format="%d")
                }
            )
            
            # Action Buttons
            st.write("")
            col_dl, col_db = st.columns(2)
            
            with col_dl:
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("Descargar Reporte (CSV)", csv, "predicciones_limpias.csv", "text/csv", type="primary", use_container_width=True)
            with col_db:
                if st.button("Archivar Proyecciones en Base de Datos", use_container_width=True):
                    with st.spinner("Escribiendo en base de datos PostgreSQL..."):
                        save_predictions_to_db(res_df)
                        st.success(f"Operacion exitosa: {len(res_df)} registros archivados.")