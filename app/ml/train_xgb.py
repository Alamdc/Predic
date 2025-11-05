import pandas as pd
import xgboost as xgb

from app.config import settings
from app.utils import (
    build_where_clause, df_from_query, feature_target_split,
    MODEL_PATH, log
)

def train_model(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int | None = None,
    edo: int | None = None,
    adm: int | None = None,
    sucursal: str | None = None,
    date_from=None,
    date_to=None,
):
    random_state = random_state if random_state is not None else settings.xgb_random_state

    where, params = build_where_clause(edo, adm, sucursal, date_from, date_to)
    sql = f"SELECT * FROM data.base_filtrada {where}"
    df = df_from_query(sql, params)

    if df.empty:
        raise ValueError("No hay datos para entrenar con los filtros proporcionados.")

    X, y = feature_target_split(df)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        tree_method="hist",
        n_jobs=0,
    )
    model.fit(X, y)

    model.save_model(MODEL_PATH)
    log.info("Modelo entrenado y guardado en %s", MODEL_PATH)
    return {"rows": len(df), "features": list(X.columns)}
