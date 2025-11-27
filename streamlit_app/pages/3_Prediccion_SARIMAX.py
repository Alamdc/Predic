import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import warnings
import json
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

# --- IMPORTACIONES LOCALES ---
from utils.db_client import fetch_data_from_api, save_sarimax_predictions_to_db
from utils.sarimax_modeling import optimize_sarimax, generate_sarimax_forecast

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Proyecciones SARIMAX", layout="wide")
warnings.filterwarnings("ignore")

# --- ESTILOS Y TÍTULO ---
st.title("Proyecciones Financieras Avanzadas")
st.markdown("### Motor Estocástico (SARIMAX / Auto-ARIMA) con Limpieza")
st.markdown("""
Este modulo utiliza modelos SARIMAX para capturar estacionalidades complejas.
Incluye pre-procesamiento con Isolation Forest para eliminar distorsiones por flujos atipicos (ej. resguardos millonarios).
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
        seasonal_period = st.selectbox("Ciclo Estacional (m)", [7, 14, 30], index=0)

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
def aplicar_limpieza_sarimax(df, col_target, contaminacion, umbral_m):
    """
    Versión simplificada del hibrido para una sola serie temporal.
    """
    g = df.copy().sort_values('fecha')
    umbral_valor = umbral_m * 1_000_000
    
    # 1. Isolation Forest
    model = IsolationForest(contamination=contaminacion, random_state=42)
    X = g[[col_target]].values
    preds = model.fit_predict(X)
    
    # 2. Regla Dura (Umbral)
    mask_exceso = g[col_target].abs() > umbral_valor
    preds[mask_exceso] = -1
    
    g['is_outlier'] = preds
    g['original'] = g[col_target]
    
    # 3. Imputar
    g.loc[g['is_outlier'] == -1, col_target] = np.nan
    g[col_target] = g[col_target].interpolate(method='linear', limit_direction='both')
    
    return g

def plot_sarimax_results(history_df, pred_df, title_text):
    fig = go.Figure()
    # Historia (Últimos 180 días visuales)
    viz_history = history_df.tail(180)
    fig.add_trace(go.Scatter(
        x=viz_history.index, y=viz_history['flujo_efectivo'],
        mode='lines', name='Historia Entrenada', line=dict(color='#6c757d', width=1.5)
    ))
    # Intervalos
    fig.add_trace(go.Scatter(
        x=pred_df['fecha_predicha'], y=pred_df['limite_superior'],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=pred_df['fecha_predicha'], y=pred_df['limite_inferior'],
        line=dict(width=0), mode='lines', fill='tonexty', fillcolor='rgba(0, 123, 255, 0.15)',
        name='Intervalo Confianza (95%)', hoverinfo='skip'
    ))
    # Predicción
    fig.add_trace(go.Scatter(
        x=pred_df['fecha_predicha'], y=pred_df['prediccion'],
        mode='lines+markers', name='Proyeccion SARIMAX',
        line=dict(color='#007bff', width=2.5), marker=dict(size=6)
    ))
    fig.update_layout(
        title=title_text, xaxis_title="Fecha", yaxis_title="Flujo ($)",
        height=450, hovermode="x unified", template="plotly_white"
    )
    return fig

# ==========================================
# 2. LÓGICA PRINCIPAL
# ==========================================

if 'search_data' not in st.session_state:
    st.session_state['search_data'] = None

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
                    st.session_state['search_data'] = None
                else:
                    df = pd.DataFrame(rows)
                    df['fecha'] = pd.to_datetime(df['fecha'])
                    df = df.sort_values('fecha')
                    st.session_state['search_data'] = df
                    st.rerun()
            except Exception as e:
                st.error(f"Error de conexion: {e}")

# --- SELECCIÓN Y ENTRENAMIENTO ---
if st.session_state['search_data'] is not None:
    df_raw = st.session_state['search_data']
    
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
    df_target = df_raw[df_raw['adm'] == adm_seleccionado].copy().set_index('fecha')
    
    # Recorte de Historia
    df_target = df_target.tail(days_history)

    # --- VISUALIZACIÓN DE LIMPIEZA PREVIA ---
    if usar_limpieza:
        with st.expander("Ver Analisis de Limpieza (Anomalias)", expanded=True):
            df_preview_clean = aplicar_limpieza_sarimax(df_target.reset_index(), 'flujo_efectivo', contamination, umbral_millones)
            n_out = (df_preview_clean['is_outlier'] == -1).sum()
            
            st.markdown(f"**Anomalias detectadas:** {n_out}")
            
            fig_clean = go.Figure()
            outliers = df_preview_clean[df_preview_clean['is_outlier'] == -1]
            fig_clean.add_trace(go.Scatter(x=outliers['fecha'], y=outliers['original'], mode='markers', name='Anomalia Eliminada', marker=dict(color='red', size=8, symbol='x')))
            fig_clean.add_trace(go.Scatter(x=df_preview_clean['fecha'], y=df_preview_clean['flujo_efectivo'], mode='lines', name='Flujo Limpio', line=dict(color='blue')))
            fig_clean.update_layout(height=350, title="Impacto de la Limpieza", template="plotly_white")
            st.plotly_chart(fig_clean, use_container_width=True)

    with col_act:
        st.write("") 
        st.write("")
        btn_entrenar = st.button("Entrenar Modelo", type="primary")

    if btn_entrenar:
        if df_target.empty:
            st.error("No hay datos suficientes despues de filtrar.")
            st.stop()
            
        sucursal_nombre = df_target['sucursal'].iloc[0] 
        
        # 1. APLICAR LIMPIEZA DEFINITIVA ANTES DE ENTRENAR
        if usar_limpieza:
            # Reseteamos indice para que la funcion trabaje bien, luego volvemos a setear
            df_clean_final = aplicar_limpieza_sarimax(df_target.reset_index(), 'flujo_efectivo', contamination, umbral_millones)
            df_train = df_clean_final.set_index('fecha')
        else:
            df_train = df_target.copy()

        # Relleno de fechas (Continuidad)
        idx_range = pd.date_range(df_train.index.min(), df_train.index.max(), freq='D')
        df_train = df_train.reindex(idx_range, fill_value=0)
        series = df_train['flujo_efectivo']

        with st.status(f"Procesando modelo para {selection}...", expanded=True) as status:
            try:
                status.write(f"Optimizando hiperparametros usando {len(series)} dias de historia...")
                
                # Entrenamos sobre la serie LIMPIA
                best_model, best_order, best_seasonal, best_aic = optimize_sarimax(
                    series, exog=None, seasonal_period=seasonal_period
                )
                
                status.write("Generando proyecciones futuras...")
                yhat, conf_df = generate_sarimax_forecast(best_model, steps=forecast_horizon)
                
                last_date = df_train.index.max()
                future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)]
                
                df_pred = pd.DataFrame({
                    "fecha_predicha": future_dates,
                    "prediccion": yhat,
                    "limite_inferior": conf_df["lower"].values,
                    "limite_superior": conf_df["upper"].values
                })
                
                st.session_state['model_result'] = {
                    "df_hist": df_train, # Guardamos la historia limpia para graficar
                    "df_pred": df_pred,
                    "meta": {
                        "aic": best_aic,
                        "order": str(best_order),
                        "seasonal": str(best_seasonal),
                        "sucursal": sucursal_nombre,
                        "adm": adm_seleccionado,
                        "edo": int(df_target['edo'].iloc[0]),
                        "last_date": last_date
                    }
                }
                status.update(label="Modelo completado exitosamente", state="complete", expanded=False)
                
            except Exception as e:
                st.error(f"Error en modelado: {e}")
                st.stop()

# --- MOSTRAR RESULTADOS ---
if 'model_result' in st.session_state:
    res = st.session_state['model_result']
    meta = res['meta']
    df_view = res['df_pred']
    
    st.divider()
    st.subheader(f"Resultados del Pronostico: {meta['adm']} - {meta['sucursal']}")
    
    # Métricas Técnicas
    k1, k2, k3 = st.columns(3)
    k1.metric("AIC (Calidad)", f"{meta['aic']:.2f}")
    k2.metric("Orden (p,d,q)", meta['order'])
    k3.metric("Estacionalidad", meta['seasonal'])
    
    # Grafico (ahora usa la historia limpia para que se vea coherente)
    st.plotly_chart(plot_sarimax_results(res['df_hist'], res['df_pred'], "Proyeccion de Flujo (Base Limpia)"), use_container_width=True)
    
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
        df_save["orden_arima"] = meta["order"]
        df_save["orden_estacional"] = meta["seasonal"]
        df_save["aic"] = meta["aic"]
        df_save["entrenado_hasta"] = meta["last_date"]
        
        if save_sarimax_predictions_to_db(df_save):
            st.success(f"Datos guardados correctamente para {meta['sucursal']}")
        else:
            st.error("Error al guardar en base de datos.")