import requests
import psycopg
import pandas as pd
from .config import API_URL, PG_DSN, PG_SCHEMA_HIST, PG_SCHEMA_PRED

def fetch_data_from_api(payload):
    """
    Obtiene datos de la API FastAPI local.
    """
    resp = requests.post(API_URL, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["rows"]

def save_predictions_to_db(df: pd.DataFrame):
    """
    Guarda el DataFrame de predicciones en PostgreSQL.
    Columnas en español coinciden con la BD.
    """
    cols = [
        "edo", "adm", "sucursal", "fecha_predicha",
        "prediccion", "limite_inferior", "limite_superior",
        "version_modelo", "entrenado_hasta", "variables_json"
    ]
    
    insert_sql = f"""
    INSERT INTO {PG_SCHEMA_PRED}.predicciones_flujo
    ({', '.join(cols)})
    VALUES ({', '.join(['%s']*len(cols))})
    ON CONFLICT (edo, adm, sucursal, fecha_predicha)
    DO UPDATE SET
      prediccion = EXCLUDED.prediccion,
      limite_inferior = EXCLUDED.limite_inferior,
      limite_superior = EXCLUDED.limite_superior,
      version_modelo = EXCLUDED.version_modelo,
      entrenado_hasta = EXCLUDED.entrenado_hasta,
      variables_json = EXCLUDED.variables_json,
      created_at = NOW();
    """
    
    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            batch = [tuple(row[c] for c in cols) for _, row in df.iterrows()]
            cur.executemany(insert_sql, batch)
            conn.commit()

def fetch_history_for_transfer(limit: int = 60):
    """
    Trae los últimos N registros históricos.
    IMPORTANTE: Trae edo, adm, sucursal.
    """
    sql = f"""
    WITH ranked AS (
        SELECT 
            edo, adm, sucursal,
            id_sucursal, -- Opcional si ya usas adm
            fecha,
            COALESCE(extsal, 0) as saldo_final,
            COALESCE(sumingresos, 0) as entradas,
            COALESCE(sumegresos, 0) as salidas,
            COALESCE(flujo_efec, 0) as flujo_efectivo,
            ROW_NUMBER() OVER (PARTITION BY edo, adm, sucursal ORDER BY fecha DESC) as rn
        FROM {PG_SCHEMA_HIST}.base_filtrada
    )
    SELECT * FROM ranked WHERE rn <= %s
    ORDER BY edo, adm, sucursal, fecha ASC
    """
    try:
        with psycopg.connect(PG_DSN) as conn:
            df = pd.read_sql(sql, conn, params=(limit,))
        return df
    except Exception as e:
        print(f"Error fetching history: {e}")
        return pd.DataFrame()

def fetch_future_predictions(days_ahead: int = 30):
    """
    Trae las predicciones futuras.
    IMPORTANTE: SELECT debe incluir edo, adm, sucursal explícitamente.
    """
    sql = f"""
    SELECT 
        edo, adm, sucursal,
        fecha_predicha as fecha,
        prediccion as flujo_efectivo
    FROM {PG_SCHEMA_PRED}.predicciones_flujo
    WHERE fecha_predicha >= CURRENT_DATE
    AND fecha_predicha <= CURRENT_DATE + interval '%s days'
    ORDER BY edo, adm, sucursal, fecha_predicha ASC
    """
    try:
        with psycopg.connect(PG_DSN) as conn:
            df = pd.read_sql(sql, conn, params=(days_ahead,))
            
            # --- CORRECCIÓN DE SEGURIDAD ---
            # Si el DataFrame viene vacío, pandas no infiere tipos.
            # Aseguramos que las columnas existan para evitar KeyError
            if df.empty:
                return pd.DataFrame(columns=["edo", "adm", "sucursal", "fecha", "flujo_efectivo"])
                
        return df
    except Exception as e:
        print(f"Error fetching predictions: {e}")
        # Retornar DF vacío con estructura correcta para evitar crash
        return pd.DataFrame(columns=["edo", "adm", "sucursal", "fecha", "flujo_efectivo"])