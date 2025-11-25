import pandas as pd
import numpy as np
from .config import RANGOS

# -------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# -------------------------------------------------------------------------

def determine_tipo_sucursal(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Calcula si es captadora (1) o pagadora (0) basado en el promedio histórico."""
    keys = ["edo", "adm", "sucursal"]
    if df_hist.empty:
        return pd.DataFrame(columns=keys + ["es_captadora"])
        
    avg_flow = df_hist.groupby(keys)["flujo_efectivo"].mean().reset_index()
    avg_flow["es_captadora"] = np.where(avg_flow["flujo_efectivo"] >= 0, 1, 0)
    return avg_flow[keys + ["es_captadora"]]

def calcular_ultimo_saldo(df: pd.DataFrame, df_tipos: pd.DataFrame) -> pd.DataFrame:
    """Obtiene el último saldo real conocido."""
    keys = ["edo", "adm", "sucursal"]
    if df.empty:
        return pd.DataFrame(columns=keys + ["saldo_final", "fecha", "tipo"])

    ultimo = (
        df.sort_values("fecha")
        .groupby(keys)
        .tail(1)[keys + ["saldo_final", "fecha"]]
        .copy()
    )
    ultimo = ultimo.merge(df_tipos, on=keys, how="left")
    ultimo["tipo"] = ultimo["es_captadora"].apply(lambda x: "captadora" if x == 1 else "pagadora")
    return ultimo

def promedios_ultimos_n(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["edo", "adm", "sucursal"]
    if df.empty:
        return pd.DataFrame(columns=keys + ["entrada_prom", "salida_prom", "flujo_efectivo_prom"])

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

    # Lógica extra: Subir rango por alta actividad
    keys = list(rangos_tipo.keys())
    if asignado in keys:
        idx = keys.index(asignado)
        if indice > 5 and idx + 1 < len(keys):
            return keys[idx + 1]
    return asignado

# -------------------------------------------------------------------------
# LÓGICA DE SIMULACIÓN (Enfoque Dominó)
# -------------------------------------------------------------------------

def simular_plan_transferencias(row, df_futuro: pd.DataFrame):
    """
    Simula el saldo día a día. Si detecta una alerta:
    1. Calcula la transferencia.
    2. Resetea el saldo al punto medio (simulando la acción).
    3. Continúa calculando los días siguientes con el saldo limpio.
    """
    tipo = row["tipo"]
    rango = row["rango"]
    saldo_simulado = float(row["saldo_final"]) 
    
    acciones = [] 
    
    if tipo not in RANGOS or rango not in RANGOS[tipo]:
        return []

    min_val, max_val = RANGOS[tipo][rango]
    punto_medio = (min_val + max_val) / 2

    # Filtro específico para esta sucursal
    filtro = (
        (df_futuro["edo"] == row["edo"]) & 
        (df_futuro["adm"] == row["adm"]) & 
        (df_futuro["sucursal"] == row["sucursal"])
    )
    
    datos = df_futuro[filtro].sort_values("fecha").copy()

    if datos.empty:
        return []

    # Bucle día a día
    for _, fila in datos.iterrows():
        flujo_diario = float(fila["flujo_efectivo"])
        fecha_actual = fila["fecha"]
        
        # 1. Aplicar flujo natural
        saldo_simulado += flujo_diario
        
        # 2. Verificar límites
        if saldo_simulado < min_val or saldo_simulado > max_val:
            # ¡ALERTA!
            transferencia = round(punto_medio - saldo_simulado, 2)
            
            acciones.append({
                "fecha": fecha_actual,
                "monto": transferencia,
                "saldo_antes": saldo_simulado,
                "saldo_despues": punto_medio
            })
            
            # 3. RESET VIRTUAL (Simulamos la corrección)
            saldo_simulado = punto_medio 
            
    return acciones

def calcular_analisis_transferencias(df_hist, df_fut):
    """Orquestador Principal."""
    if df_hist.empty:
        return pd.DataFrame()

    # --- 1. LIMPIEZA DE DATOS (Tipos) ---
    for df in [df_hist, df_fut]:
        if not df.empty:
            if "edo" in df.columns:
                df["edo"] = pd.to_numeric(df["edo"], errors='coerce').fillna(0).astype(int)
            if "adm" in df.columns:
                df["adm"] = pd.to_numeric(df["adm"], errors='coerce').fillna(0).astype(int)
            if "sucursal" in df.columns:
                df["sucursal"] = df["sucursal"].astype(str).str.strip()

    # --- 2. ESTADO ACTUAL ---
    df_tipos = determine_tipo_sucursal(df_hist)
    ult = calcular_ultimo_saldo(df_hist, df_tipos)
    prom = promedios_ultimos_n(df_hist)
    
    keys = ["edo", "adm", "sucursal"]
    df = pd.merge(ult, prom, on=keys, how="left")
    
    # Asignar Rango
    df["rango"] = df.apply(asignar_rango, axis=1)
    
    # --- 3. SIMULACIÓN ---
    columna_planes = df.apply(
        lambda r: simular_plan_transferencias(r, df_fut), axis=1
    )
    
    # --- 4. EXTRACCIÓN PRIMERA ALERTA ---
    def extraer_primera_alerta(lista_acciones):
        if not lista_acciones:
            return None, 0.0
        # Retorna fecha y monto de la primera acción
        return lista_acciones[0]["fecha"], lista_acciones[0]["monto"]

    datos_desempaquetados = columna_planes.apply(extraer_primera_alerta)
    
    df["fecha_alerta"] = [x[0] for x in datos_desempaquetados]
    df["monto_sugerido"] = [x[1] for x in datos_desempaquetados]
    
    return df