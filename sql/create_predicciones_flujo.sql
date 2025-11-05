CREATE SCHEMA IF NOT EXISTS data;

CREATE TABLE IF NOT EXISTS data.predicciones_flujo (
    edo SMALLINT,
    adm INTEGER,
    sucursal VARCHAR(200),
    fecha_predicha DATE,
    yhat NUMERIC(18,2) NOT NULL,
    yhat_lower NUMERIC(18,2),
    yhat_upper NUMERIC(18,2),
    modelo_version VARCHAR(50) DEFAULT 'xgb_v1',
    trained_until DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    features JSONB,
    PRIMARY KEY (edo, adm, sucursal, fecha_predicha)
);

CREATE INDEX IF NOT EXISTS idx_predicciones_lookup
  ON data.predicciones_flujo (edo, adm, sucursal, fecha_predicha);
