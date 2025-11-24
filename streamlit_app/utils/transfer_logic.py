import pandas as pd
import numpy as np
import streamlit as st # Importamos st para poder mostrar errores si es necesario
from .config import RANGOS

def determine_tipo_sucursal(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Calcula si es captadora/pagadora basado en histórico."""
    keys = ["edo", "adm", "sucursal"]
    avg_flow = df_hist.groupby(keys)["flujo_efectivo"].mean().reset_index()
    avg_flow["es_captadora"] = np.where(avg_flow["flujo_efectivo"] >= 0, 1, 0)
    return avg_flow[keys + ["es_captadora"]]

def calcular_ultimo_saldo(df: pd.DataFrame, df_tipos: pd.DataFrame) -> pd.DataFrame:
    keys = ["edo", "adm", "sucursal"]
    # Tomamos el último registro ordenado por fecha
    ultimo = (
        df.sort_values("fecha")
        .groupby(keys)
        .tail(1)[keys + ["saldo_final", "fecha"]] # Incluimos fecha para referencia
        .copy()
    )
    ultimo = ultimo.merge(df_tipos, on=keys, how="left")
    ultimo["tipo"] = ultimo["es_captadora"].apply(lambda x: "captadora" if x == 1 else "pagadora")
    return ultimo

def promedios_ultimos_n(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["edo", "adm", "sucursal"]
    prom = (
        df.groupby(keys)
        .agg({
            "entradas": "mean",
            "salidas": "mean",
            "flujo_efectivo": "mean",
        })
        .reset_index()
    ).rename(columns={
        "entradas": "entrada_prom",
        "salidas": "salida_prom",
        "flujo_efectivo": "flujo_efectivo_prom",
    })
    prom["tamaño_flujo"] = prom["entrada_prom"] + prom["salida_prom"]
    denom = prom["flujo_efectivo_prom"] + 1e-6
    prom["indice_actividad"] = abs(prom["tamaño_flujo"] / denom)
    return prom

def asignar_rango(row) -> str:
    tipo = row.get("tipo", "pagadora")
    tam_flujo = row.get("tamaño_flujo", 0)
    indice = row.get("indice_actividad", 0)

    rangos_tipo = RANGOS.get(tipo, {})
    asignado = None
    
    for rango, (lo, hi) in rangos_tipo.items():
        if lo <= tam_flujo <= hi:
            asignado = rango
            break

    if asignado is None:
        if "E" in rangos_tipo and tam_flujo > rangos_tipo["E"][1]:
            return "E"
        return "Fuera de Rango"

    keys = list(rangos_tipo.keys())
    if asignado in keys:
        idx = keys.index(asignado)
        if indice > 5 and idx + 1 < len(keys):
            return keys[idx + 1]
    return asignado

def encontrar_dia_transferencia(row, df_futuro: pd.DataFrame):
    """
    Simula el saldo día a día.
    """
    tipo = row["tipo"]
    rango = row["rango"]
    # Convertimos a float nativo para evitar problemas de tipos numpy
    saldo = float(row["saldo_final"]) 
    
    if tipo not in RANGOS or rango not in RANGOS[tipo]:
        return None, 0.0

    min_val, max_val = RANGOS[tipo][rango]
    punto_medio = (min_val + max_val) / 2

    # --- FILTRO ROBUSTO ---
    # Usamos .values para asegurar comparación segura, 
    # asumiendo que ya limpiamos los tipos en la función principal
    filtro = (
        (df_futuro["edo"] == row["edo"]) & 
        (df_futuro["adm"] == row["adm"]) & 
        (df_futuro["sucursal"] == row["sucursal"])
    )
    
    # Ordenamos por fecha para simular cronológicamente
    datos = df_futuro[filtro].sort_values("fecha").copy()

    # DEBUG: Si no encuentra datos para esta sucursal, retornamos vacío
    if datos.empty:
        return None, 0.0

    fecha_salida = None
    
    # Simulación día a día
    for _, fila in datos.iterrows():
        flujo_predicho = float(fila["flujo_efectivo"])
        saldo += flujo_predicho
        
        # Chequeo de límites
        if saldo < min_val or saldo > max_val:
            fecha_salida = fila["fecha"]
            break

    # Cálculo monto sugerido
    suma_total_futuro = float(datos["flujo_efectivo"].sum())
    saldo_proyectado_final = float(row["saldo_final"]) + suma_total_futuro
    
    # Si nunca se sale en los días predichos, tomamos la última fecha disponible
    # pero el monto sugerido sería 0 si el saldo final está en rango.
    if fecha_salida is None:
        # Chequeamos si al final del periodo estamos fuera
        if saldo_proyectado_final < min_val or saldo_proyectado_final > max_val:
             fecha_salida = datos["fecha"].max()
        else:
             return None, 0.0 # Todo en orden
    
    transferencia = round(punto_medio - saldo_proyectado_final, 2)
    return fecha_salida, transferencia

def calcular_analisis_transferencias(df_hist, df_fut):
    """Función orquestadora principal con LIMPIEZA DE DATOS."""
    if df_hist.empty:
        return pd.DataFrame()

    # --- 0. PRE-LIMPIEZA DE TIPOS (CRUCIAL) ---
    # Convertimos edo y adm a enteros para garantizar el cruce
    # Limpiamos espacios en blanco en sucursales
    for df in [df_hist, df_fut]:
        if not df.empty:
            df["edo"] = pd.to_numeric(df["edo"], errors='coerce').fillna(0).astype(int)
            df["adm"] = pd.to_numeric(df["adm"], errors='coerce').fillna(0).astype(int)
            if "sucursal" in df.columns:
                df["sucursal"] = df["sucursal"].astype(str).str.strip()

    # 1. Determinar tipos
    df_tipos = determine_tipo_sucursal(df_hist)
    
    # 2. Obtener estado actual
    ult = calcular_ultimo_saldo(df_hist, df_tipos)
    
    # 3. Obtener métricas promedio
    prom = promedios_ultimos_n(df_hist)
    
    # 4. Unir todo
    keys = ["edo", "adm", "sucursal"]
    df = pd.merge(ult, prom, on=keys, how="left")
    
    # 5. Asignar Rango
    df["rango"] = df.apply(asignar_rango, axis=1)
    
    # 6. Calcular fechas de transferencia
    # Pasamos el df_fut completo, el filtrado ocurre dentro fila por fila
    resultados = df.apply(
        lambda r: encontrar_dia_transferencia(r, df_fut), axis=1
    )
    
    # Desempaquetar tuplas
    df["fecha_alerta"] = [x[0] for x in resultados]
    df["monto_sugerido"] = [x[1] for x in resultados]
    
    return df