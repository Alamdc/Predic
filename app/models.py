from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Date, Numeric, SmallInteger, TIMESTAMP

Base = declarative_base()

class BaseFiltrada(Base):
    __tablename__ = "base_filtrada"
    __table_args__ = {"schema": "data"}

    edo = Column(SmallInteger, primary_key=True)
    adm = Column(Integer, primary_key=True)
    sucursal = Column(String(200))
    fecha = Column(Date, primary_key=True)
    dia = Column(SmallInteger)
    entradas = Column(Numeric(18, 2))
    salidas = Column(Numeric(18, 2))
    flujo_efectivo = Column(Numeric(18, 2))
    dia_semana = Column(SmallInteger)
    semana_anio = Column(SmallInteger)
    mes = Column(SmallInteger)
    anio = Column(SmallInteger)
    trimestre = Column(SmallInteger)
    media_movil_3 = Column(Numeric(18, 2))
    media_movil_5 = Column(Numeric(18, 2))
    media_movil_10 = Column(Numeric(18, 2))
    media_movil_14 = Column(Numeric(18, 2))
    lag_1 = Column(Numeric(18, 2))
    lag_2 = Column(Numeric(18, 2))
    lag_3 = Column(Numeric(18, 2))
    lag_5 = Column(Numeric(18, 2))
    processed_at = Column(TIMESTAMP)

class PrediccionFlujo(Base):
    __tablename__ = "predicciones_flujo"
    __table_args__ = {"schema": "data"}

    # Sin PK estricta: tabla de “hechos”/histórico
    edo = Column(SmallInteger)
    adm = Column(Integer)
    sucursal = Column(String(200))
    fecha = Column(Date)
    flujo_real = Column(Numeric(18, 2))
    flujo_predicho = Column(Numeric(18, 2))
    created_at = Column(TIMESTAMP)
