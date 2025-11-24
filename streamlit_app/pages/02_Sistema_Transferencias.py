import streamlit as st
import pandas as pd
from utils.db_client import fetch_history_for_transfer, fetch_future_predictions
from utils.transfer_logic import calcular_analisis_transferencias

st.set_page_config(page_title="Sistema de Transferencias", layout="wide")
st.title("Sistema de Alertas de Transferencia")

st.markdown("""
Esta herramienta analiza el **saldo histórico real** y las **predicciones futuras (XGBoost)**
para determinar si una sucursal se saldrá de sus rangos de efectivo operativos.
""")

# --- CONTROLES ---
with st.sidebar:
    st.header("Configuración de Análisis")
    
    n_history = st.slider("Días de historia (para promedios)", 15, 90, 30)
    n_future = st.slider("Días de horizonte (predicción)", 7, 60, 21)
    
    st.divider()
    btn_run = st.button("Ejecutar Análisis", type="primary")

# --- LÓGICA PRINCIPAL ---
if btn_run:
    with st.spinner("Consultando Base de Datos..."):
        # 1. Traer datos
        df_hist = fetch_history_for_transfer(limit=n_history)
        df_fut = fetch_future_predictions(days_ahead=n_future)
        
        if df_hist.empty:
            st.error("No se encontraron datos históricos en `tdv_data.base_filtrada`.")
            st.stop()
        
        if df_fut.empty:
            st.warning("No hay predicciones futuras en `predicciones_flujo`. El análisis usará solo historia (incompleto).")
    
    with st.spinner("Calculando rangos y simulaciones..."):
        # 2. Procesar lógica de negocio
        resultado = calcular_analisis_transferencias(df_hist, df_fut)
    
    # --- VISUALIZACIÓN ---
    st.success(f"Análisis completado para {len(resultado)} sucursales.")
    
    # KPI Totales
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Sucursales", len(resultado))
    
    # Filtrar las que requieren acción (monto != 0 y fecha no nula)
    accion_requerida = resultado[resultado["monto_sugerido"].abs() > 1000].copy() # Filtro ruido < 1000
    kpi2.metric("Alertas Activas", len(accion_requerida))
    
    total_mov = accion_requerida["monto_sugerido"].abs().sum()
    kpi3.metric("Volumen Sugerido Movimiento", f"${total_mov:,.2f}")
    
    st.divider()
    
    # Tabla Principal
    st.subheader("Detalle por Sucursal")
    
    # Filtros visuales
    filtro_tipo = st.multiselect("Filtrar por Tipo", ["captadora", "pagadora"], default=["captadora", "pagadora"])
    if filtro_tipo:
        resultado = resultado[resultado["tipo"].isin(filtro_tipo)]
        
    cols_visual = [
        "edo", "adm", "sucursal", "tipo", "rango", 
        "saldo_final", "flujo_efectivo_prom", 
        "fecha_alerta", "monto_sugerido"
    ]
    
    # Formateo visual
    st.dataframe(
        resultado[cols_visual].sort_values("fecha_alerta"),
        use_container_width=True,
        column_config={
            "saldo_final": st.column_config.NumberColumn("Saldo Actual", format="$%.2f"),
            "flujo_efectivo_prom": st.column_config.NumberColumn("Flujo Prom", format="$%.2f"),
            "monto_sugerido": st.column_config.NumberColumn(
                "Transferencia Sugerida", 
                format="$%.2f",
                help="Negativo: Retirar dinero. Positivo: Enviar dinero."
            ),
            "fecha_alerta": st.column_config.DateColumn("Fecha Límite"),
            "rango": "Rango Asignado"
        }
    )
    
    # Descarga
    csv = resultado.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Descargar Reporte Completo (CSV)",
        data=csv,
        file_name="reporte_transferencias.csv",
        mime="text/csv"
    )

else:
    st.info("Configura los parámetros en la barra lateral y presiona 'Ejecutar Análisis'.")