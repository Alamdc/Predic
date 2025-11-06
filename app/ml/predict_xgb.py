import pandas as pd
import xgboost as xgb

from app.utils import (
    build_where_clause, df_from_query, feature_target_split,
    insert_predictions, to_records_for_api, MODEL_PATH, log, exists_model
)
# Nota: Ya no importamos 'train_model' aquí.

def predict_and_upload(
    edo: int | None = None,
    adm: int | None = None,
    sucursal: str | None = None,
    date_from=None,
    date_to=None,
):
    """
    Carga el modelo existente, genera predicciones y las sube a la BD.
    
    Lanza un FileNotFoundError si el modelo no existe.
    """
    
    # 1. Verificación Rápida: Si el modelo no existe, fallar inmediatamente.
    if not exists_model(MODEL_PATH):
        log.warning("Se intentó predecir, pero el modelo no existe en %s", MODEL_PATH)
        # Lanzamos un error específico que la API (main.py) puede atrapar.
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}. Ejecute /train primero.")

    # 2. Obtener los datos para la predicción
    log.info("Obteniendo datos para predecir...")
    where, params = build_where_clause(edo, adm, sucursal, date_from, date_to)
    sql = f"SELECT * FROM data.base_filtrada {where}"
    df = df_from_query(sql, params)

    if df.empty:
        log.info("No se encontraron datos para predecir con los filtros dados.")
        return {"inserted": 0, "preview": []}

    # 3. Preparar datos y cargar modelo
    log.info("Preparando datos y cargando modelo...")
    X, _ = feature_target_split(df)  # Asumimos que feature_target_split maneja el modo 'predict'
    
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    # 4. Predecir y subir a la base de datos
    log.info("Generando predicciones...")
    df["prediccion_flujo"] = model.predict(X)
    
    log.info("Insertando predicciones en la base de datos...")
    insert_predictions(df)  # Confiamos en tu helper de utils.py

    # 5. Preparar la respuesta
    preview = to_records_for_api(df) # Confiamos en tu helper de utils.py
    log.info("Predicciones generadas e insertadas: %d filas", len(df))
    return {"inserted": len(df), "preview": preview}