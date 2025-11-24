import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()

# Actualizamos credenciales y el esquema por defecto a 'tdv_data'
PG_DSN = f"host={os.getenv('PG_HOST')} port={os.getenv('PG_PORT')} dbname={os.getenv('PG_DATABASE')} user={os.getenv('PG_USER')} password={os.getenv('PG_PASSWORD')}"
PG_SCHEMA = os.getenv('PG_SCHEMA', 'tdv_data') 

app = FastAPI(title="Datos base_filtrada API (tdv_data)", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataQuery(BaseModel):
    edo: Optional[int] = None
    adm: Optional[int] = None
    sucursal: Optional[str] = None
    start: Optional[str] = None  # YYYY-MM-DD
    end: Optional[str] = None    # YYYY-MM-DD
    limit: int = 100000

@app.post("/data")
async def get_data(q: DataQuery):
    where = []
    params = {}

    if q.edo is not None:
        where.append("edo = %(edo)s")
        params['edo'] = q.edo
    if q.adm is not None:
        where.append("adm = %(adm)s")
        params['adm'] = q.adm
    if q.sucursal:
        where.append("sucursal = %(sucursal)s")
        params['sucursal'] = q.sucursal
    if q.start:
        where.append("fecha >= %(start)s")
        params['start'] = q.start
    if q.end:
        where.append("fecha <= %(end)s")
        params['end'] = q.end

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    # SQL Actualizado para tdv_data.base_filtrada
    # Mapeamos las columnas nuevas a los nombres que espera el modelo (entradas, salidas, etc.)
    # Calculamos las fechas (anio, mes, trimestre) al vuelo con EXTRACT
    sql = f"""
    SELECT 
        edo, adm, sucursal, fecha,
        EXTRACT(DAY FROM fecha) as dia,
        sumingresos as entradas, 
        sumegresos as salidas, 
        flujo_efec as flujo_efectivo,
        EXTRACT(YEAR FROM fecha) as anio,
        EXTRACT(QUARTER FROM fecha) as trimestre,
        EXTRACT(MONTH FROM fecha) as mes,
        EXTRACT(WEEK FROM fecha) as semana_anio,
        EXTRACT(ISODOW FROM fecha) as dia_semana,
        media_movil_3, media_movil_5, media_movil_10, media_movil_14,
        lag_1, lag_2, lag_3, lag_5
    FROM {PG_SCHEMA}.base_filtrada
    {where_sql}
    ORDER BY edo, adm, sucursal, fecha
    LIMIT %(limit)s
    """
    params['limit'] = q.limit

    try:
        with psycopg.connect(PG_DSN, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                return {"rows": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))