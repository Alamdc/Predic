import os
import logging
from contextlib import contextmanager
from typing import Optional, Tuple, Sequence, Dict, Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.database import engine as _engine
from app.config import settings

# Logging básico
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("predicciones")

MODEL_PATH = settings.xgb_model_path
NON_FEATURE_COLS = {"sucursal", "fecha", "processed_at", "flujo_efectivo"}

def get_engine() -> Engine:
    return _engine

@contextmanager
def db_connect(engine: Optional[Engine] = None):
    eng = engine or get_engine()
    conn = eng.connect()
    try:
        yield conn
    finally:
        conn.close()

def df_from_query(sql: str, params: Optional[Dict[str, Any]] = None,
                  engine: Optional[Engine] = None) -> pd.DataFrame:
    eng = engine or get_engine()
    with db_connect(eng) as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

def run_sql(sql: str, params: Optional[Dict[str, Any]] = None,
            engine: Optional[Engine] = None) -> None:
    eng = engine or get_engine()
    with db_connect(eng) as conn:
        trans = conn.begin()
        try:
            conn.execute(text(sql), params or {})
            trans.commit()
        except SQLAlchemyError as e:
            trans.rollback()
            log.exception("DB error")
            raise e

def safe_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(df["flujo_efectivo"], errors="coerce")
    drop_cols = [c for c in NON_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)
    X = safe_numeric(X, X.columns)
    return X, y

def ensure_predictions_table(engine: Optional[Engine] = None) -> None:
    create_sql = """
    CREATE TABLE IF NOT EXISTS data.predicciones_flujo (
        edo SMALLINT,
        adm INTEGER,
        sucursal VARCHAR(200),
        fecha DATE,
        flujo_real NUMERIC(18,2),
        flujo_predicho NUMERIC(18,2),
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    run_sql(create_sql, engine=engine)

def insert_predictions(df_pred: pd.DataFrame, engine: Optional[Engine] = None) -> None:
    eng = engine or get_engine()
    ensure_predictions_table(eng)
    tmp = df_pred.rename(
        columns={"flujo_efectivo": "flujo_real", "prediccion_flujo": "flujo_predicho"}
    )[["edo", "adm", "sucursal", "fecha", "flujo_real", "flujo_predicho"]].copy()

    for c in ("flujo_real", "flujo_predicho"):
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp.to_sql("predicciones_flujo", eng, schema="data", if_exists="append", index=False)
    log.info("Insertadas %d filas en data.predicciones_flujo", len(tmp))

def build_where_clause(
    edo: Optional[int],
    adm: Optional[int],
    sucursal: Optional[str],
    date_from: Optional[pd.Timestamp],
    date_to: Optional[pd.Timestamp]
) -> Tuple[str, Dict[str, Any]]:
    clauses = []
    params: Dict[str, Any] = {}
    if edo is not None:
        clauses.append("edo = :edo")
        params["edo"] = edo
    if adm is not None:
        clauses.append("adm = :adm")
        params["adm"] = adm
    if sucursal:
        clauses.append("sucursal = :sucursal")
        params["sucursal"] = sucursal
    if date_from is not None:
        clauses.append("fecha >= :date_from")
        params["date_from"] = pd.to_datetime(date_from).date()
    if date_to is not None:
        clauses.append("fecha <= :date_to")
        params["date_to"] = pd.to_datetime(date_to).date()

    where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
    return where_sql, params

def exists_model(path: Optional[str] = None) -> bool:
    return os.path.exists(path or MODEL_PATH)

def to_records_for_api(df: pd.DataFrame) -> list:
    cols = ["edo", "adm", "sucursal", "fecha", "flujo_efectivo", "prediccion_flujo"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].head(200).copy()
    if "fecha" in out.columns:
        out["fecha"] = pd.to_datetime(out["fecha"]).dt.date
    return out.to_dict(orient="records")
