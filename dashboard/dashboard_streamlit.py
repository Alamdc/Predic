import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import requests
from dotenv import load_dotenv 

load_dotenv()
API_URL = "http://localhost:8000"  


DB_URL = os.getenv("DATABASE_URL")

st.set_page_config(page_title="Predicciones de Flujo", layout="wide")
st.title("Predicciones de Flujo de Efectivo")

engine = create_engine(DB_URL)

st.sidebar.header("Acciones manuales")

if st.sidebar.button("Entrenar modelo"):
    with st.spinner("Entrenando modelo..."):
        res = requests.post(f"{API_URL}/train")
    st.success(res.json().get("message", "Entrenamiento completado"))

if st.sidebar.button("Generar predicciones"):
    with st.spinner("Generando predicciones..."):
        res = requests.post(f"{API_URL}/predict")
    st.success(res.json().get("message", "Predicciones completadas"))

st.sidebar.header("Filtros")
limit = st.sidebar.number_input("Límites de registros", min_value=50, max_value=5000, value=500, step=50)

df = pd.read_sql(
    f"SELECT * FROM data.predicciones_flujo ORDER BY fecha DESC LIMIT {int(limit)}",
    engine
)

if df.empty:
    st.info("No hay datos en data.predicciones_flujo todavía. Ejecuta /predict.")
else:
    st.subheader("Tabla reciente")
    st.dataframe(df)

    st.subheader("Serie temporal")
    if {"fecha", "flujo_real", "flujo_predicho"}.issubset(df.columns):
        plot_df = df.sort_values("fecha").set_index("fecha")[["flujo_real", "flujo_predicho"]]
        st.line_chart(plot_df)

    st.subheader("Error absoluto (si hay flujo_real)")
    if "flujo_real" in df.columns and df["flujo_real"].notna().any():
        tmp = df.dropna(subset=["flujo_real"]).copy()
        tmp["abs_error"] = (tmp["flujo_real"] - tmp["flujo_predicho"]).abs()
        st.bar_chart(tmp.set_index("fecha")["abs_error"])
