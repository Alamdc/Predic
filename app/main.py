from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.utils import df_from_query, exists_model, log
from app.schemas import (
    TrainRequest, PredictRequest, MessageResponse, HealthResponse,
    PredictionsResponse, PredictionRecord
)
from app.ml.train_xgb import train_model
from app.ml.predict_xgb import predict_and_upload

app = FastAPI(title=settings.api_title)

@app.get("/", response_model=MessageResponse)
def home():
    return {"message": "API de predicciones de flujo de efectivo (XGBoost)"}

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        df = df_from_query("SELECT COUNT(*) AS c FROM data.base_filtrada")
        cnt = int(df.iloc[0]["c"])
    except Exception:
        cnt = -1
    return HealthResponse(model_loaded=exists_model(), rows_available=cnt)

@app.post("/train", response_model=MessageResponse)
def train(body: TrainRequest):
    try:
        info = train_model(**body.dict())
        return {"message": f"Modelo entrenado. Filas: {info['rows']} | Features: {len(info['features'])}"}
    except Exception as e:
        log.exception("Error en /train")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", response_model=MessageResponse)
def predict(body: PredictRequest):
    try:
        res = predict_and_upload(**body.dict())
        return {"message": f"Predicciones insertadas: {res['inserted']}"}
    except Exception as e:
        log.exception("Error en /predict")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predictions", response_model=PredictionsResponse)
def get_predictions(limit: int = 100):
    try:
        df = df_from_query(
            "SELECT edo, adm, sucursal, fecha, flujo_real, flujo_predicho "
            "FROM data.predicciones_flujo ORDER BY fecha DESC LIMIT :lim",
            params={"lim": limit}
        )
        items = [
            PredictionRecord(
                edo=int(r.edo), adm=int(r.adm),
                sucursal=str(r.sucursal), fecha=r.fecha,
                flujo_real=float(r.flujo_real) if r.flujo_real is not None else None,
                flujo_predicho=float(r.flujo_predicho)
            )
            for r in df.itertuples(index=False)
        ]
        return PredictionsResponse(count=len(items), items=items)
    except Exception as e:
        log.exception("Error en /predictions")
        raise HTTPException(status_code=400, detail=str(e))
