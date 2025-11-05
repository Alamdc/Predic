import pandas as pd
import xgboost as xgb

from app.utils import (
    build_where_clause, df_from_query, feature_target_split,
    insert_predictions, to_records_for_api, MODEL_PATH, log, exists_model
)
from app.ml.train_xgb import train_model

def predict_and_upload(
    edo: int | None = None,
    adm: int | None = None,
    sucursal: str | None = None,
    date_from=None,
    date_to=None,
    retrain_quick: bool = False
):
    # Opcional: reentrenar rápido con los filtros (si el usuario lo pide)
    if retrain_quick or not exists_model(MODEL_PATH):
        train_model(
            n_estimators=200, learning_rate=0.07, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            edo=edo, adm=adm, sucursal=sucursal, date_from=date_from, date_to=date_to
        )

    where, params = build_where_clause(edo, adm, sucursal, date_from, date_to)
    sql = f"SELECT * FROM data.base_filtrada {where}"
    df = df_from_query(sql, params)

    if df.empty:
        return {"inserted": 0, "preview": []}

    X, _ = feature_target_split(df)

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    df["prediccion_flujo"] = model.predict(X)
    insert_predictions(df)

    preview = to_records_for_api(df)
    log.info("Predicciones generadas: %d filas", len(df))
    return {"inserted": len(df), "preview": preview}
