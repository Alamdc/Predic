import streamlit as st

# Configuración de página
st.set_page_config(page_title="Inicio - Gestión de Efectivo", layout="wide")

# Encabezado Principal
st.title("Sistema Integral de Predicción y Gestión de Efectivo")
st.markdown("### Plataforma de Inteligencia para Tesorería")
st.divider()

# Introducción Ejecutiva
st.markdown("""
Bienvenido a la plataforma centralizada para la administración del flujo de efectivo. 
Esta solución tecnológica integra modelos de aprendizaje automático con reglas de negocio financieras para optimizar la logística de valores en la red de sucursales.

El sistema permite anticipar necesidades de liquidez y automatizar la toma de decisiones logísticas.
""")
st.write("") # Espacio vertical

# Sección de Módulos (Diseño de Tarjetas)
st.subheader("Arquitectura de la Solución")

col1, col2 = st.columns(2, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown("#### 1. Generador de Predicciones")
        st.markdown("**Motor de Proyección (XGBoost)**")
        st.markdown("""
        Este módulo es el responsable de la inteligencia predictiva. Utiliza datos históricos para entrenar modelos personalizados por sucursal.
        
        * **Conexión:** Extrae datos depurados .
        * **Capacidad:** Genera escenarios a corto y mediano plazo .
        * **Tecnología:** Algoritmos de Gradient Boosting con intervalos de confianza.
        """)

with col2:
    with st.container(border=True):
        st.markdown("#### 2. Sistema de Transferencias")
        st.markdown("**Motor de Simulación Logística**")
        st.markdown("""
        Este módulo transforma las predicciones en acciones operativas. Simula el comportamiento diario del saldo en bóveda.
        
        * **Análisis:** Detecta rupturas en los límites operativos.
        * **Acción:** Sugiere montos exactos de envío o recolección.
        * **Lógica:** Simulación de compensación automática.
        """)

# Sección de Estado del Sistema (Técnico)
st.write("")
st.subheader("Estado de Conectividad")

col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    with st.container(border=True):
        st.markdown("**Fuente de Datos**")
        st.caption("API Gateway")
        st.caption("Estado: Conectado")

with col_t2:
    with st.container(border=True):
        st.markdown("**Almacenamiento**")
        st.caption("PostgreSQL")
        st.caption("Lectura/Escritura Habilitada")

with col_t3:
    with st.container(border=True):
        st.markdown("**Modelado**")
        st.caption("Librería: XGBoost")
        st.caption("Modo: Inferencia Dinámica")

st.divider()
st.info("Para iniciar una operación, despliegue el menú lateral izquierdo y seleccione el módulo deseado.")