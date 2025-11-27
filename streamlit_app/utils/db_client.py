import requests
import psycopg
import pandas as pd
from datetime import date
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
    Asume que la tabla en BD tiene los mismos nombres de columnas en español que el DF.
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
    Trae los últimos N registros históricos por sucursal.
    Construye el DataFrame manualmente para evitar warnings de SQLAlchemy.
    """
    sql = f"""
    WITH ranked AS (
        SELECT 
            edo, adm, sucursal,
            id_sucursal, 
            fecha,
            COALESCE(extsal, 0) as saldo_final,
            COALESCE(sumingresos, 0) as entradas,
            COALESCE(sumegresos, 0) as salidas,
            COALESCE(flujo_efec, 0) as flujo_efectivo,
            ROW_NUMBER() OVER (PARTITION BY edo, adm, sucursal ORDER BY fecha DESC) as rn
        FROM {PG_SCHEMA_HIST}.base_filtrada
    )
    SELECT 
        edo, adm, sucursal, id_sucursal, fecha, 
        saldo_final, entradas, salidas, flujo_efectivo 
    FROM ranked WHERE rn <= %s
    ORDER BY edo, adm, sucursal, fecha ASC
    """
    try:
        with psycopg.connect(PG_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()
                
                # Obtener nombres de columnas del cursor
                cols = [desc[0] for desc in cur.description]
                return pd.DataFrame(rows, columns=cols)
                
    except Exception as e:
        print(f"Error fetching history: {e}")
        return pd.DataFrame()

def fetch_future_predictions(start_date_str: str = None, days_ahead: int = 30):
    """
    Trae las predicciones futuras a partir de una fecha dada.
    
    Args:
        start_date_str: Fecha de inicio (YYYY-MM-DD). Si es None, usa hoy.
                        Importante para simular con datos antiguos (2022).
        days_ahead: Cuántos días a futuro traer.
    """
    if not start_date_str:
        start_date_str = date.today().isoformat()

    sql = f"""
    SELECT 
        edo, adm, sucursal,
        fecha_predicha as fecha,
        prediccion as flujo_efectivo
    FROM {PG_SCHEMA_PRED}.predicciones_flujo
    WHERE fecha_predicha > %s 
    AND fecha_predicha <= %s::date + interval '%s days'
    ORDER BY edo, adm, sucursal, fecha_predicha ASC
    """
    try:
        with psycopg.connect(PG_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (start_date_str, start_date_str, days_ahead))
                rows = cur.fetchall()
                
                cols = [desc[0] for desc in cur.description]
                df = pd.DataFrame(rows, columns=cols)
                
                if df.empty:
                     # Estructura vacía para evitar crash
                     return pd.DataFrame(columns=["edo", "adm", "sucursal", "fecha", "flujo_efectivo"])
                return df
                
    except Exception as e:
        print(f"Error fetching predictions: {e}")
        return pd.DataFrame(columns=["edo", "adm", "sucursal", "fecha", "flujo_efectivo"])
    

def save_sarimax_predictions_to_db(df: pd.DataFrame):
    """
    Guarda las predicciones generadas por SARIMAX en PostgreSQL.
    """
    cols = [
        "edo", "adm", "sucursal", "fecha_predicha",
        "prediccion", "limite_inferior", "limite_superior",
        "orden_arima", "orden_estacional", "aic", 
        "entrenado_hasta"
    ]
    
    # Query de inserción con UPSERT (On Conflict Update)
    insert_sql = f"""
    INSERT INTO {PG_SCHEMA_PRED}.predicciones_sarimax
    ({', '.join(cols)})
    VALUES ({', '.join(['%s']*len(cols))})
    ON CONFLICT (edo, adm, sucursal, fecha_predicha)
    DO UPDATE SET
      prediccion = EXCLUDED.prediccion,
      limite_inferior = EXCLUDED.limite_inferior,
      limite_superior = EXCLUDED.limite_superior,
      orden_arima = EXCLUDED.orden_arima,
      orden_estacional = EXCLUDED.orden_estacional,
      aic = EXCLUDED.aic,
      entrenado_hasta = EXCLUDED.entrenado_hasta,
      created_at = NOW();
    """
    
    try:
        with psycopg.connect(PG_DSN) as conn:
            with conn.cursor() as cur:
                # Convertimos el DF a lista de tuplas respetando el orden de cols
                batch = []
                for _, row in df.iterrows():
                    batch.append(tuple(row.get(c, None) for c in cols))
                
                cur.executemany(insert_sql, batch)
                conn.commit()
        return True
    except Exception as e:
        print(f"Error guardando SARIMAX: {e}")
        return False