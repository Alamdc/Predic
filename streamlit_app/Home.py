import streamlit as st

st.set_page_config(page_title="Home - Flujo de Efectivo", layout="wide")

st.title("Sistema de Predicciones de Flujo de Efectivo")

st.markdown("""
Bienvenido al sistema de predicciones.

Este aplicativo permite:
1. Conectarse a la API de datos filtrados (`fastapi_app`).
2. Entrenar modelos XGBoost dinámicamente por agrupación (Edo/Adm/Sucursal).
3. Generar proyecciones financieras con intervalos de confianza.
4. Guardar resultados en base de datos PostgreSQL.

👈 **Selecciona 'Generador Predicciones' en el menú lateral para comenzar.**
""")