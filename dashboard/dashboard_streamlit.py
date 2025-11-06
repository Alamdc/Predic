# En: dashboard/dashboard_streamlit.py
# Ya no necesitamos os, load_dotenv, ni create_engine.
import pandas as pd
import streamlit as st
import requests  # Solo necesitamos requests
import altair as alt

# URL de tu API de FastAPI (asumiendo que corre en el puerto 8000)
API_BASE_URL = "http://127.0.0.1:8000"

# --- 1. Funciones para llamar a la API ---

@st.cache_data(ttl=60)  # Cachear por 60 segundos
def fetch_predictions(limit: int = 100):
    """Obtiene las últimas predicciones desde el endpoint /predictions."""
    try:
        res = requests.get(f"{API_BASE_URL}/predictions", params={"limit": limit})
        res.raise_for_status()  # Lanza un error si la API falla
        data = res.json()
        
        # Convertimos los 'items' del JSON directamente a un DataFrame
        df = pd.DataFrame(data["items"])
        
        # Convertir la columna de fecha
        if not df.empty:
            df["fecha"] = pd.to_datetime(df["fecha"])
        return df
        
    except requests.ConnectionError:
        st.error("No se pudo conectar a la API. ¿Ejecutaste 'uvicorn app.main:app'?")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar predicciones: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_health():
    """Verifica el estado de la API desde el endpoint /health."""
    try:
        res = requests.get(f"{API_BASE_URL}/health")
        if res.status_code == 200:
            return res.json()
    except requests.ConnectionError:
        return None # API no disponible
    return None

# --- 2. Configuración de la Página de Streamlit ---

st.set_page_config(page_title="Predicciones de Flujo", layout="wide")
st.title("💸 Predicciones de Flujo de Efectivo")


# --- 3. Barra Lateral (Sidebar) ---

with st.sidebar:
    st.header("Acciones manuales")

    if st.button("Entrenar modelo"):
        try:
            with st.spinner("Entrenando modelo... (esto puede tardar)"):
                res = requests.post(f"{API_BASE_URL}/train", json={}) # pasamos un body vacío
            res.raise_for_status()
            st.success(res.json().get("message", "Entrenamiento completado"))
            st.cache_data.clear() # Limpiamos la caché para ver datos nuevos
            st.rerun()
        except Exception as e:
            st.error(f"Error de entrenamiento: {e.json().get('detail', 'Error')}")

    if st.button("Generar predicciones"):
        try:
            with st.spinner("Generando predicciones..."):
                # Enviamos un body vacío para que use los filtros por defecto
                res = requests.post(f"{API_BASE_URL}/predict", json={})
            res.raise_for_status()
            st.success(res.json().get("message", "Predicciones completadas"))
            st.cache_data.clear() # Limpiamos la caché para ver datos nuevos
            st.rerun()
        except Exception as e:
            st.error(f"Error de predicción: {e.json().get('detail', 'Error')}")
    
    st.sidebar.header("Filtros")
    limit = st.sidebar.number_input(
        "Límite de registros", min_value=50, max_value=5000, value=500, step=50
    )
    
    st.header("Estado de la API")
    health_data = fetch_health()
    if health_data:
        st.success("API Conectada")
        if health_data.get("model_loaded"):
            st.info("✅ Modelo Cargado")
        else:
            st.warning("⚠️ Modelo NO Cargado. (Presiona 'Entrenar modelo')")
        st.caption(f"Filas en DB: {health_data.get('rows_available')}")
    else:
        st.error("API Desconectada.")

# --- 4. Cuerpo Principal del Dashboard ---

# <--- CAMBIO PRINCIPAL: Ya no usamos pd.read_sql ---
df = fetch_predictions(limit=limit)
# <--- FIN DEL CAMBIO ---

if df.empty:
    st.info("No hay datos en `data.predicciones_flujo` todavía. Intenta 'Generar predicciones' desde la barra lateral.")
else:
    st.subheader("Tabla reciente")
    st.dataframe(df)

    st.subheader("Serie temporal de Flujo (Agregado por día)")
    if {"fecha", "flujo_real", "flujo_predicho"}.issubset(df.columns):
        # Agrupamos por día para que el gráfico sea más limpio
        plot_df = df.groupby(pd.Grouper(key="fecha", freq="D")).agg(
            flujo_real=("flujo_real", "sum"),
            flujo_predicho=("flujo_predicho", "sum")
        ).reset_index()
        
        # Convertimos a formato "largo" para que Altair funcione mejor
        plot_df_melted = plot_df.melt('fecha', var_name='Tipo de Flujo', value_name='Monto')

        chart = alt.Chart(plot_df_melted).mark_line(point=True).encode(
            x=alt.X("fecha", title="Fecha"),
            y=alt.Y("Monto", title="Monto de Flujo"),
            color=alt.Color("Tipo de Flujo", title="Tipo de Flujo"),
            tooltip=["fecha", "Tipo de Flujo", "Monto"]
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Error absoluto (si hay flujo_real)")
    if "flujo_real" in df.columns and df["flujo_real"].notna().any():
        tmp = df.dropna(subset=["flujo_real"]).copy()
        tmp["abs_error"] = (tmp["flujo_real"] - tmp["flujo_predicho"]).abs()
        
        st.bar_chart(tmp.set_index("fecha")["abs_error"])