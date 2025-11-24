import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# ==========================================
# 1. CONFIGURACIÓN DE API (FastAPI)
# ==========================================
API_HOST = os.getenv('API_HOST', '127.0.0.1')
API_PORT = int(os.getenv('API_PORT', '8000'))
API_URL = f"http://{API_HOST}:{API_PORT}/data"

# ==========================================
# 2. CONFIGURACIÓN DE BASE DE DATOS (PostgreSQL)
# ==========================================
PG_HOST = os.getenv('PG_HOST')
PG_PORT = os.getenv('PG_PORT')
PG_DATABASE = os.getenv('PG_DATABASE')
PG_USER = os.getenv('PG_USER')
PG_PASSWORD = os.getenv('PG_PASSWORD')

# Cadena de conexión (DSN) para psycopg
PG_DSN = (
    f"host={PG_HOST} port={PG_PORT} dbname={PG_DATABASE} "
    f"user={PG_USER} password={PG_PASSWORD}"
)

# Definición de Esquemas
# tdv_data: Esquema donde vive la historia real (base_filtrada)
# tdv_data: Esquema donde guardaremos las predicciones (predicciones_flujo)
PG_SCHEMA_HIST = "tdv_data"
PG_SCHEMA_PRED = "tdv_data"

# ==========================================
# 3. REGLAS DE NEGOCIO (RANGOS)
# ==========================================
# Definición de límites operativos de efectivo por tipo de sucursal
RANGOS = {
    "captadora": {
        "A": (10000, 35000),
        "B": (15000, 50000),
        "C": (25000, 80000),
        "D": (40000, 140000),
        "E": (60000, 300000),
    },
    "pagadora": {
        "A": (15000, 40000),
        "B": (30000, 80000),
        "C": (60000, 160000),
        "D": (120000, 300000),
        "E": (240000, 600000),
    },
}