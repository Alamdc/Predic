from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.utils import df_from_query, exists_model, log
from app.schemas import (
    TrainRequest, PredictRequest, MessageResponse, HealthResponse,
    PredictionsResponse, PredictionRecord
)
from app.ml.train_xgb import train_model
from app.ml.predict_xgb import predict_and_upload # Esta es la versión refinada

app = FastAPI(title=settings.api_title)

@app.get("/", response_model=MessageResponse)
def home():
    return {"message": "API de predicciones de flujo de efectivo (XGBoost)"}

@app.get("/health", response_model=HealthResponse)
def health():
    """
    Verifica la salud de la API, la conexión a la BD y si el modelo existe.
    """
    try:
        # Prueba la conexión a la BD
        df = df_from_query("SELECT COUNT(*) AS c FROM data.base_filtrada")
        cnt = int(df.iloc[0]["c"])
    except Exception:
        cnt = -1  # Indica un error de conexión o de tabla
        
    return HealthResponse(model_loaded=exists_model(), rows_available=cnt)

@app.post("/train", response_model=MessageResponse)
def train(body: TrainRequest):
    """
    Endpoint para entrenar (o re-entrenar) el modelo de ML.
    """
    try:
        log.info("Solicitud /train recibida.")
        # Pasamos todos los argumentos del body a tu función de entrenamiento
        info = train_model(**body.dict())
        return {"message": f"Modelo entrenado. Filas: {info['rows']} | Features: {len(info['features'])}"}
    except ValueError as ve:
        # Error común si no hay datos
        log.warning("Error en /train: %s", ve)
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        log.exception("Error inesperado en /train")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=MessageResponse)
def predict(body: PredictRequest):
    """
    Endpoint para generar y guardar predicciones.
    Requiere que el modelo ya haya sido entrenado.
    """
    try:
        log.info("Solicitud /predict recibida.")
        
        # Obtenemos los argumentos del body de Pydantic
        request_data = body.dict()
        
        # IMPORTANTE: Eliminamos 'retrain_quick' si es que aún existe en tu
        # schema 'PredictRequest', ya que la nueva función no lo acepta.
        if 'retrain_quick' in request_data:
            del request_data['retrain_quick']
            
        # Llamamos a la función de predicción SOLAMENTE con los filtros
        res = predict_and_upload(**request_data)
        
        return {"message": f"Predicciones insertadas: {res['inserted']}"}
    
    except FileNotFoundError as e:
        # ¡Aquí atrapamos el error! Si predict_and_upload no encuentra el modelo.
        log.warning("Error en /predict: Modelo no encontrado.")
        raise HTTPException(status_code=404, detail=str(e)) # 404 Not Found
        
    except Exception as e:
        log.exception("Error inesperado en /predict")
        # 400 Bad Request es bueno si los datos de entrada son malos
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predictions", response_model=PredictionsResponse)
def get_predictions(limit: int = 100):
    """
    Endpoint para que el dashboard consuma las predicciones ya guardadas.
    """
    try:
        df = df_from_query(
            "SELECT edo, adm, sucursal, fecha, flujo_real, flujo_predicho "
            "FROM data.predicciones_flujo ORDER BY fecha DESC LIMIT :lim",
            params={"lim": limit}
        )
        
        # Convertimos el DataFrame a una lista de Pydantic Schemas
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
        raise HTTPException(status_code=500, detail=str(e))