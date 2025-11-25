import streamlit as st
import pandas as pd
from utils.db_client import fetch_history_for_transfer, fetch_future_predictions
from utils.transfer_logic import calcular_analisis_transferencias

# Configuración de página minimalista y profesional
st.set_page_config(page_title="Gestión de Efectivo", layout="wide")

# Título principal sin adornos
st.title("Sistema de Gestión de Efectivo y Transferencias")
st.markdown("### Panel de Control de Tesorería")
st.markdown("""
Este módulo utiliza modelos predictivos para simular el comportamiento del flujo de efectivo. 
El sistema identifica desviaciones en los rangos operativos y sugiere movimientos logísticos para optimizar el saldo en sucursales.
""")
st.divider()

# --- BARRA LATERAL (FILTROS) ---
with st.sidebar:
    st.header("Parámetros de Simulación")
    
    with st.container(border=True):
        n_history = st.slider("Histórico Base (Días)", 15, 90, 30, help="Días pasados utilizados para calcular promedios y tendencias.")
        n_future = st.slider("Horizonte de Proyección (Días)", 7, 30, 7, help="Días a futuro para simular el comportamiento del saldo.")
    
    st.write("") # Espacio
    btn_run = st.button("Ejecutar Análisis", type="primary", use_container_width=True)

# --- LÓGICA PRINCIPAL ---
if btn_run:
    # 1. Carga de Datos
    with st.spinner("Consultando base de datos histórica..."):
        df_hist = fetch_history_for_transfer(limit=n_history)
        
        if df_hist.empty:
            st.error("Error: No se encontraron datos históricos en la base de datos.")
            st.stop()
            
        # Detección de fecha de corte
        max_hist_date = pd.to_datetime(df_hist["fecha"]).max()
        start_date_str = max_hist_date.strftime("%Y-%m-%d")
        
    with st.spinner("Cargando proyecciones de flujo de efectivo..."):
        df_fut = fetch_future_predictions(start_date_str=start_date_str, days_ahead=n_future)
        
        if df_fut.empty:
            st.warning(f"Atención: No existen predicciones posteriores al {start_date_str}. Por favor ejecute el generador de predicciones.")
            st.stop()

    # 2. Motor de Simulación
    with st.spinner("Procesando algoritmo de simulación..."):
        try:
            resultado = calcular_analisis_transferencias(df_hist, df_fut)
        except Exception as e:
            st.error(f"Error crítico en el cálculo: {e}")
            st.stop()
    
    # 3. Segmentación de Resultados
    # Filtro: Existe fecha sugerida Y el monto es material (mayor a 500 pesos)
    alertas = resultado[
        (resultado["fecha_alerta"].notnull()) & 
        (resultado["monto_sugerido"].abs() > 500)
    ].copy()

    # --- DASHBOARD (VISTA PROFESIONAL) ---
    
    # Sección de KPIs en tarjetas
    st.subheader("Resumen Ejecutivo")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        with st.container(border=True):
            st.metric(label="Sucursales Analizadas", value=len(resultado))
    
    with col_kpi2:
        with st.container(border=True):
            count_alertas = len(alertas)
            st.metric(label="Requieren Atención", value=count_alertas, delta=None)

    with col_kpi3:
        with st.container(border=True):
            # Suma de retiros (dinero que regresa a bóveda central)
            retiros = alertas[alertas["monto_sugerido"] < 0]["monto_sugerido"].sum()
            st.metric(label="Retiros Sugeridos", value=f"${abs(retiros):,.0f}")

    with col_kpi4:
        with st.container(border=True):
            # Suma de envíos (dinero que sale a sucursales)
            envios = alertas[alertas["monto_sugerido"] > 0]["monto_sugerido"].sum()
            st.metric(label="Envíos Sugeridos", value=f"${envios:,.0f}")

    st.write("") # Espacio vertical

    # Sistema de Pestañas para organizar la información
    tab1, tab2 = st.tabs(["Plan de Operaciones (Alertas)", "Detalle General de la Red"])

    # --- PESTAÑA 1: LO URGENTE ---
    with tab1:
        if not alertas.empty:
            st.info("Las siguientes sucursales proyectan un desborde de sus límites operativos en el horizonte analizado.")
            
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                tipo_filtro = st.multiselect("Filtrar por Tipo", ["captadora", "pagadora"], default=["captadora", "pagadora"])
            
            # Aplicar filtro
            alertas_view = alertas[alertas["tipo"].isin(tipo_filtro)].sort_values("fecha_alerta")

            # Selección de columnas para reporte limpio
            cols_mostrar = [
                "edo", "adm", "sucursal", "tipo", "rango", 
                "saldo_final", "fecha_alerta", "monto_sugerido"
            ]
            
            st.dataframe(
                alertas_view[cols_mostrar],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "edo": st.column_config.NumberColumn("Estado", format="%d"),
                    "adm": st.column_config.NumberColumn("ID Sucursal", format="%d"),
                    "sucursal": "Sucursal",
                    "tipo": "Perfil",
                    "rango": "Nivel",
                    "saldo_final": st.column_config.NumberColumn("Saldo Actual", format="$ %.2f"),
                    "fecha_alerta": st.column_config.DateColumn("Fecha Sugerida de Operación", format="DD/MM/YYYY"),
                    "monto_sugerido": st.column_config.NumberColumn(
                        "Monto Transferencia", 
                        format="$ %.2f",
                        help="Positivo: Enviar efectivo a sucursal. Negativo: Recolección de efectivo."
                    )
                }
            )
            
            # Botón de descarga principal
            csv = alertas_view.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Descargar Plan Operativo (CSV)", 
                data=csv, 
                file_name="plan_operativo_transferencias.csv", 
                mime="text/csv",
                type="primary"
            )
        else:
            st.success("El análisis no detectó desviaciones operativas para el periodo seleccionado. No se requieren movimientos.")

    # --- PESTAÑA 2: TODO EL UNIVERSO ---
    with tab2:
        st.write("Listado completo del estado actual de todas las sucursales analizadas.")
        
        st.dataframe(
            resultado,
            use_container_width=True,
            hide_index=True,
            column_config={
                "monto_sugerido": st.column_config.NumberColumn("Transferencia (Si aplica)", format="$ %.2f"),
                "saldo_final": st.column_config.NumberColumn("Saldo Actual", format="$ %.2f"),
                "entrada_prom": st.column_config.NumberColumn("Entrada Prom.", format="$ %.2f"),
                "salida_prom": st.column_config.NumberColumn("Salida Prom.", format="$ %.2f"),
                "flujo_efectivo_prom": st.column_config.NumberColumn("Flujo Neto Prom.", format="$ %.2f"),
                "indice_actividad": st.column_config.ProgressColumn(
                    "Índice Actividad", 
                    format="%.2f", 
                    min_value=0, 
                    max_value=10,
                    help="Relación entre volumen total y flujo neto."
                ),
            }
        )

else:
    # Pantalla de bienvenida (Estado inicial)
    st.info("Configure los parámetros en el panel lateral y presione 'Ejecutar Análisis' para comenzar.")