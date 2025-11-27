-- Crear tabla en el esquema tdv_data
CREATE TABLE IF NOT EXISTS tdv_data.predicciones_sarimax (
    edo SMALLINT,
    adm INTEGER,
    sucursal VARCHAR(200),
    fecha_predicha DATE,
    
    -- Predicciones
    prediccion NUMERIC(18,2) NOT NULL,
    limite_inferior NUMERIC(18,2),
    limite_superior NUMERIC(18,2),
    
    -- Metadatos del modelo SARIMAX
    orden_arima VARCHAR(50),      -- Ej: (1, 1, 1)
    orden_estacional VARCHAR(50), -- Ej: (0, 1, 1, 7)
    aic NUMERIC(18, 2),           -- Métrica de calidad del ajuste
    
    entrenado_hasta DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (edo, adm, sucursal, fecha_predicha)
);

CREATE INDEX IF NOT EXISTS idx_predicciones_sarimax_lookup
  ON tdv_data.predicciones_sarimax (edo, adm, sucursal, fecha_predicha);