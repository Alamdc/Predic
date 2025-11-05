from typing import Optional, List
from datetime import date
from pydantic import BaseModel, Field, ConfigDict

class TrainRequest(BaseModel):
    n_estimators: int = Field(300, ge=50, le=5000)
    learning_rate: float = Field(0.05, gt=0, le=1)
    max_depth: int = Field(6, ge=1, le=20)
    subsample: float = Field(0.8, gt=0, le=1)
    colsample_bytree: float = Field(0.8, gt=0, le=1)
    random_state: int | None = None
    edo: Optional[int] = None
    adm: Optional[int] = None
    sucursal: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None

class PredictRequest(BaseModel):
    edo: Optional[int] = None
    adm: Optional[int] = None
    sucursal: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    retrain_quick: bool = False

class MessageResponse(BaseModel):
    message: str

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    rows_available: int

class PredictionRecord(BaseModel):
    edo: int
    adm: int
    sucursal: str
    fecha: date
    flujo_real: float | None = None
    flujo_predicho: float

class PredictionsResponse(BaseModel):
    count: int
    items: List[PredictionRecord]

class HealthResponse(BaseModel):
    # Permite campos que empiezan con "model_"
    model_config = ConfigDict(protected_namespaces=())

    status: str = "ok"
    model_loaded: bool
    rows_available: int