# Predicciones de Flujo con XGBoost + Streamlit + FastAPI (solo datos)

## Estructura
```
predicciones_flujo/
├── .env.example
├── requirements.txt
├── README.md
├── fastapi_app/
│   └── main.py
├── streamlit_app/
│   └── app.py
└── sql/
    └── create_predicciones_flujo.sql
```

## Pasos
1) Crea la tabla de predicciones (ver `sql/create_predicciones_flujo.sql`).  
2) Copia `.env.example` a `.env` y llena tus credenciales.  
3) Instala dependencias:
```bash
pip install -r requirements.txt
```
4) Levanta FastAPI (terminal 1):
```bash
uvicorn fastapi_app.main:app --host $API_HOST --port $API_PORT --reload
```
5) Levanta Streamlit (terminal 2):
```bash
streamlit run streamlit_app/app.py
```
6) En la UI de Streamlit: **Cargar datos** → define **Días a predecir** → **Guardar predicciones**.
