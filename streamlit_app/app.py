import os
import json
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import date, timedelta
from dotenv import load_dotenv
import psycopg
import streamlit as st

load_dotenv()

# Config
API_HOST = os.getenv('API_HOST', '127.0.0.1')
API_PORT = int(os.getenv('API_PORT', '8000'))
API_URL = f"http://{API_HOST}:{API_PORT}/data"

PG_DSN = f"host={os.getenv('PG_HOST')} port={os.getenv('PG_PORT')} dbname={os.getenv('PG_DATABASE')} user={os.getenv('PG_USER')} password={os.getenv('PG_PASSWORD')}"
PG_SCHEMA = os.getenv('PG_SCHEMA', 'data')

st.set_page_config(page_title="Predicciones de Flujo – XGBoost", layout="wide")
st.title("Predicciones de Flujo de Efectivo (XGBoost)")
st.caption("FastAPI solo para datos • Entrenamiento y predicción ocurren en Streamlit")

# Sidebar – Filtros
with st.sidebar:
    st.header("Filtros de extracción (FastAPI)")
    edo = st.number_input("edo", min_value=0, value=None, step=1, format="%d")
    adm = st.number_input("adm", min_value=0, value=None, step=1, format="%d")
    sucursal = st.text_input("sucursal")
    start = st.date_input("Desde", value=None)
    end = st.date_input("Hasta", value=None)

    if start and end and start > end:
        st.error("La fecha 'Desde' no puede ser mayor que 'Hasta'.")

    n_days = st.number_input("Días a predecir", min_value=1, max_value=60, value=14)
    test_size_days = st.number_input("Días recientes para validar (RMSE)", min_value=7, max_value=120, value=28)

    if st.button("Cargar datos de FastAPI"):
        payload = {
            "edo": int(edo) if edo is not None else None,
            "adm": int(adm) if adm is not None else None,
            "sucursal": sucursal if sucursal else None,
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
            "limit": 1_000_000
        }
        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()["rows"]
        st.session_state["raw_rows"] = data
        st.success(f"Se cargaron {len(data)} filas.")

# Datos
rows = st.session_state.get("raw_rows")
if not rows:
    st.info("Usa la barra lateral para traer datos desde FastAPI.")
    st.stop()

df = pd.DataFrame(rows)
if df.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# Vista rápida
st.subheader("Histórico cargado")
st.dataframe(df.head(50), use_container_width=True)

# Definiciones de columnas
BASE_FEATURES = [
    "entradas", "salidas",
    "anio", "trimestre", "mes", "semana_anio", "dia_semana",
    "media_movil_3", "media_movil_5", "media_movil_10", "media_movil_14",
    "lag_1", "lag_2", "lag_3", "lag_5"
]
TARGET = "flujo_efectivo"

# Tipos
if "fecha" in df.columns:
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.date

results = []
for (g_edo, g_adm, g_suc), g in df.groupby(["edo", "adm", "sucursal"], dropna=False):
    g = g.sort_values("fecha").reset_index(drop=True)

    cutoff = len(g) - int(test_size_days)
    if cutoff < 30:
        st.warning(f"Grupo ({g_edo}, {g_adm}, {g_suc}) con pocos datos (n={len(g)}). Se omite.")
        continue

    train_df = g.iloc[:cutoff].copy()
    test_df = g.iloc[cutoff:].copy()

    X_train = train_df[BASE_FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[BASE_FEATURES]
    y_test = test_df[TARGET]

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_pred_val - y_test.values) ** 2)))
    resid_std = float(np.std((y_test.values - y_pred_val)))

    last_date = g['fecha'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, int(n_days) + 1)]

    history = g[["fecha", TARGET, "entradas", "salidas"]].copy().sort_values("fecha")
    s_flujo = list(history[TARGET].values)

    median_entradas = float(history["entradas"].median()) if not history["entradas"].isna().all() else 0.0
    median_salidas = float(history["salidas"].median()) if not history["salidas"].isna().all() else 0.0

    preds_rows = []

    def cal_features_from_date(d: date):
        iso = pd.Timestamp(d).isocalendar()
        return {
            "anio": d.year,
            "trimestre": (d.month - 1)//3 + 1,
            "mes": d.month,
            "semana_anio": int(iso.week),
            "dia_semana": d.weekday() + 1
        }

    def moving_avgs(series):
        s = pd.Series(series)
        mv3 = float(s.tail(3).mean()) if len(s) >= 1 else 0.0
        mv5 = float(s.tail(5).mean()) if len(s) >= 1 else 0.0
        mv10 = float(s.tail(10).mean()) if len(s) >= 1 else 0.0
        mv14 = float(s.tail(14).mean()) if len(s) >= 1 else 0.0
        return mv3, mv5, mv10, mv14

    for d in future_dates:
        lag1 = float(s_flujo[-1]) if len(s_flujo) >= 1 else 0.0
        lag2 = float(s_flujo[-2]) if len(s_flujo) >= 2 else lag1
        lag3 = float(s_flujo[-3]) if len(s_flujo) >= 3 else lag2
        lag5 = float(s_flujo[-5]) if len(s_flujo) >= 5 else lag3

        mv3, mv5, mv10, mv14 = moving_avgs(s_flujo)
        cal = cal_features_from_date(d)

        xrow = {
            "entradas": median_entradas,
            "salidas": median_salidas,
            **cal,
            "media_movil_3": mv3,
            "media_movil_5": mv5,
            "media_movil_10": mv10,
            "media_movil_14": mv14,
            "lag_1": lag1,
            "lag_2": lag2,
            "lag_3": lag3,
            "lag_5": lag5,
        }

        x_df = pd.DataFrame([xrow])[BASE_FEATURES]
        yhat = float(model.predict(x_df)[0])

        yhat_lower = yhat - 1.96 * resid_std
        yhat_upper = yhat + 1.96 * resid_std

        preds_rows.append({
            "edo": g_edo,
            "adm": g_adm,
            "sucursal": g_suc,
            "fecha_predicha": d,
            "yhat": yhat,
            "yhat_lower": yhat_lower,
            "yhat_upper": yhat_upper,
            "modelo_version": "xgb_v1",
            "trained_until": last_date,
            "features": json.dumps(xrow)
        })

        s_flujo.append(yhat)

    pred_df = pd.DataFrame(preds_rows)
    pred_df["rmse_val"] = rmse

    results.append(((g_edo, g_adm, g_suc), pred_df))

st.subheader("Resultados de predicción")
all_preds = []
for key, pred_df in results:
    g_edo, g_adm, g_suc = key
    st.markdown(f"**Grupo:** edo={g_edo} • adm={g_adm} • sucursal={g_suc}")
    st.dataframe(pred_df[["fecha_predicha","yhat","yhat_lower","yhat_upper"]], use_container_width=True)
    all_preds.append(pred_df)

all_preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

if not all_preds_df.empty and st.button("Subir Predicciones"):
    try:
        cols = [
            "edo","adm","sucursal","fecha_predicha",
            "yhat","yhat_lower","yhat_upper",
            "modelo_version","trained_until","features"
        ]
        insert_sql = f"""
        INSERT INTO {PG_SCHEMA}.predicciones_flujo
        ({', '.join(cols)})
        VALUES ({', '.join(['%s']*len(cols))})
        ON CONFLICT (edo, adm, sucursal, fecha_predicha)
        DO UPDATE SET
          yhat = EXCLUDED.yhat,
          yhat_lower = EXCLUDED.yhat_lower,
          yhat_upper = EXCLUDED.yhat_upper,
          modelo_version = EXCLUDED.modelo_version,
          trained_until = EXCLUDED.trained_until,
          features = EXCLUDED.features,
          created_at = NOW();
        """
        with psycopg.connect(PG_DSN) as conn:
            with conn.cursor() as cur:
                batch = [tuple(row[c] for c in cols) for _, row in all_preds_df.iterrows()]
                cur.executemany(insert_sql, batch)
                conn.commit()
        st.success(f"Guardadas {len(all_preds_df)} filas en {PG_SCHEMA}.predicciones_flujo")
    except Exception as e:
        st.error(f"Error al guardar: {e}")

if not all_preds_df.empty:
    csv = all_preds_df.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar predicciones", data=csv, file_name="predicciones_flujo.csv", mime="text/csv")

st.caption("© predicciones – XGBoost + Streamlit + FastAPI (solo datos)")
