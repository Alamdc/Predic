import os
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import streamlit as st
import psycopg

from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterSampler

# ===================== Carga de variables =====================
load_dotenv()

API_HOST = os.getenv('API_HOST', '127.0.0.1')
API_PORT = int(os.getenv('API_PORT', '8000'))
API_URL = f"http://{API_HOST}:{API_PORT}/data"

PG_DSN = (
    f"host={os.getenv('PG_HOST')} "
    f"port={os.getenv('PG_PORT')} "
    f"dbname={os.getenv('PG_DATABASE')} "
    f"user={os.getenv('PG_USER')} "
    f"password={os.getenv('PG_PASSWORD')}"
)
PG_SCHEMA = os.getenv('PG_SCHEMA', 'data')

# ===================== Config Streamlit =====================
st.set_page_config(page_title="Predicciones de Flujo – XGBoost+", layout="wide")
st.title("Predicciones de Flujo de Efectivo (XGBoost+)")

# ===================== Helper compatible con distintas versiones de XGBoost =====================
def safe_fit(model, Xtr, ytr, Xva=None, yva=None, early_stopping=0):
    """
    Entrena XGBRegressor probando distintas firmas de .fit() para ser
    compatible con versiones que no aceptan callbacks, eval_metric o
    early_stopping_rounds en fit.
    """
    fit_kwargs = {}
    if Xva is not None and yva is not None:
        fit_kwargs["eval_set"] = [(Xva, yva)]

    # 1) Intentar con callbacks (algunas builds de 2.x)
    if early_stopping and Xva is not None:
        try:
            cb = xgb.callback.EarlyStopping(rounds=int(early_stopping), min_delta=0.0, save_best=True)
            model.fit(Xtr, ytr, verbose=False, callbacks=[cb], **fit_kwargs)
            return model
        except TypeError:
            pass
        except Exception:
            pass

    # 2) Intentar con early_stopping_rounds en fit (1.7.x y algunas builds)
    if early_stopping and Xva is not None:
        try:
            model.fit(Xtr, ytr, verbose=False, early_stopping_rounds=int(early_stopping), **fit_kwargs)
            return model
        except TypeError:
            pass
        except Exception:
            pass

    # 3) Intento simple con verbose=False
    try:
        model.fit(Xtr, ytr, verbose=False, **fit_kwargs)
        return model
    except TypeError:
        pass
    except Exception:
        pass

    # 4) Último recurso: sin verbose ni extras
    model.fit(Xtr, ytr, **fit_kwargs)
    return model

# ===================== Sidebar =====================
with st.sidebar:
    st.header("Filtros de extracción (FastAPI)")
    edo = st.number_input("edo (opcional)", min_value=0, value=None, step=1, format="%d")
    adm = st.number_input("adm (opcional)", min_value=0, value=None, step=1, format="%d")
    sucursal = st.text_input("sucursal (opcional)")
    start = st.date_input("Desde (opcional)", value=None)
    end = st.date_input("Hasta (opcional)", value=None)

    if start and end and start > end:
        st.error("La fecha 'Desde' no puede ser mayor que 'Hasta'.")

    st.divider()
    st.subheader("Horizonte y validación")
    n_days = st.number_input("Días a predecir", min_value=1, max_value=90, value=21)
    test_size_days = st.number_input("Días para validación simple (últimos N)", min_value=7, max_value=180, value=28)
    n_rolling_folds = st.number_input("Folds de backtest rolling", min_value=1, max_value=10, value=3)

    st.divider()
    st.subheader("Tuning por ADM (aleatorio)")
    do_tune = st.checkbox("Activar tuning por ADM", value=True)
    n_param_samples = st.number_input("Número de combinaciones a probar", min_value=5, max_value=100, value=20)
    early_stopping = st.number_input("Early stopping rounds", min_value=0, max_value=300, value=50)

    st.divider()
    st.subheader("Escenarios Entradas/Salidas futuras")
    escenario_e = st.slider("Ajuste Entradas futuras (%)", min_value=-50, max_value=50, value=0, step=5)
    escenario_s = st.slider("Ajuste Salidas futuras (%)", min_value=-50, max_value=50, value=0, step=5)

    if st.button("Cargar datos de FastAPI"):
        payload = {
            "edo": int(edo) if edo is not None else None,
            "adm": int(adm) if adm is not None else None,
            "sucursal": sucursal if sucursal else None,
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
            "limit": 1_000_000
        }
        resp = requests.post(API_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()["rows"]
        st.session_state["raw_rows"] = data
        st.success(f"Se cargaron {len(data)} filas.")

# ===================== Datos =====================
rows = st.session_state.get("raw_rows")
if not rows:
    st.info("Usa la barra lateral para aplicar filtros")
    st.stop()

df = pd.DataFrame(rows)
if df.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

if "fecha" in df.columns:
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.date

# ===================== Features =====================
BASE_FEATURES = [
    "entradas", "salidas",
    "anio", "trimestre", "mes", "semana_anio", "dia_semana",
    "media_movil_3", "media_movil_5", "media_movil_10", "media_movil_14",
    "lag_1", "lag_2", "lag_3", "lag_5"
]
TARGET = "flujo_efectivo"

def enrich_features(g: pd.DataFrame):
    """Features adicionales 'al vuelo' sin tocar DB."""
    g = g.copy()
    g["neto_es"] = (g["entradas"] - g["salidas"]).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        g["ratio_es"] = np.where(
            (g["salidas"].astype(float) + 1.0) != 0.0,
            g["entradas"].astype(float) / (g["salidas"].astype(float) + 1.0),
            0.0
        )
    g["abs_diff_es"] = (g["entradas"].astype(float) - g["salidas"].astype(float)).abs()
    g["sin_dow"] = np.sin(2*np.pi*(g["dia_semana"].astype(float)/7.0))
    g["cos_dow"] = np.cos(2*np.pi*(g["dia_semana"].astype(float)/7.0))
    g["sin_month"] = np.sin(2*np.pi*(g["mes"].astype(float)/12.0))
    g["cos_month"] = np.cos(2*np.pi*(g["mes"].astype(float)/12.0))
    extra = ["neto_es","ratio_es","abs_diff_es","sin_dow","cos_dow","sin_month","cos_month"]
    return g, BASE_FEATURES + extra

def build_future_calendar(d: date) -> dict:
    ts = pd.Timestamp(d).isocalendar()
    return {
        "anio": d.year,
        "trimestre": (d.month - 1)//3 + 1,
        "mes": d.month,
        "semana_anio": int(ts.week),
        "dia_semana": d.weekday() + 1
    }

def compute_estacional_tables(g: pd.DataFrame):
    """Medianas por (dow, mes) para estimar futuras entradas/salidas."""
    g = g.copy()
    g["dow"] = g["dia_semana"]
    g["m"] = g["mes"]
    e_tab = g.pivot_table(index="dow", columns="m", values="entradas", aggfunc="median")
    s_tab = g.pivot_table(index="dow", columns="m", values="salidas", aggfunc="median")
    e_global = float(np.nanmedian(g["entradas"])) if "entradas" in g else 0.0
    s_global = float(np.nanmedian(g["salidas"])) if "salidas" in g else 0.0
    return e_tab, s_tab, e_global, s_global

def get_future_es(dow:int, month:int, e_tab, s_tab, e_global:float, s_global:float, adj_e:float, adj_s:float):
    e = e_tab.loc[dow, month] if (dow in e_tab.index and month in e_tab.columns) else np.nan
    s = s_tab.loc[dow, month] if (dow in s_tab.index and month in s_tab.columns) else np.nan
    e = float(e) if not np.isnan(e) else e_global
    s = float(s) if not np.isnan(s) else s_global
    e = e * (1.0 + adj_e/100.0)
    s = s * (1.0 + adj_s/100.0)
    return e, s

def rolling_backtest(series_df: pd.DataFrame, features: list, target: str, n_folds:int=3, params:dict=None, early_stopping:int=50):
    """Backtest con origen rodante en espacio transformado (si se usa)."""
    if params is None:
        params = dict(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.0, reg_lambda=1.0,
            random_state=42, n_jobs=4, objective="reg:squarederror"
        )
    n = len(series_df)
    fold_size = max(7, n // (n_folds + 1))
    rmses, mapes = [], []

    for k in range(1, n_folds+1):
        split = k * fold_size
        if split >= n-7:
            break
        train = series_df.iloc[:split]
        test = series_df.iloc[split:split+fold_size]
        if len(test) < 7:
            continue

        Xtr = train[features]; ytr = train[target]
        Xte = test[features]; yte = test[target]

        params_bt = {**params, "eval_metric": "rmse"}
        model = xgb.XGBRegressor(**params_bt)
        safe_fit(model, Xtr, ytr, Xva=Xte, yva=yte, early_stopping=early_stopping)

        pred = model.predict(Xte)
        rmses.append(np.sqrt(mean_squared_error(yte, pred)))
        mape = mean_absolute_percentage_error(np.where(yte==0, 1e-6, yte), pred)
        mapes.append(mape)

    return {
        "folds": len(rmses),
        "rmse_mean": float(np.mean(rmses)) if rmses else None,
        "rmse_std": float(np.std(rmses)) if rmses else None,
        "mape_mean": float(np.mean(mapes)) if mapes else None,
        "mape_std": float(np.std(mapes)) if mapes else None,
    }

def random_tune(Xtr, ytr, Xte, yte, n_samples:int=20, early_stopping:int=50, seed:int=42):
    """Búsqueda aleatoria de hiperparámetros compatible con XGBoost sklearn."""
    space = {
        "n_estimators": [200, 300, 400, 600, 800],
        "max_depth": [3,4,5,6,7,8,9],
        "learning_rate": np.logspace(np.log10(0.01), np.log10(0.2), 10),
        "subsample": np.linspace(0.6, 1.0, 9),
        "colsample_bytree": np.linspace(0.6, 1.0, 9),
        "min_child_weight": np.linspace(1.0, 10.0, 10),
        "reg_alpha": np.linspace(0.0, 2.0, 9),
        "reg_lambda": np.linspace(0.1, 5.0, 10),
    }
    rng = np.random.RandomState(seed)
    sampler = list(ParameterSampler(space, n_iter=n_samples, random_state=rng))

    best_rmse = float("inf")
    best = None
    for p in sampler:
        params = dict(
            objective="reg:squarederror",
            n_jobs=4,
            random_state=42,
            eval_metric="rmse",  # en el constructor
            **p
        )
        model = xgb.XGBRegressor(**params)
        safe_fit(model, Xtr, ytr, Xva=Xte, yva=yte, early_stopping=early_stopping)

        pred = model.predict(Xte)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best = params

    return best if best is not None else dict(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.0, reg_lambda=1.0,
        random_state=42, n_jobs=4, objective="reg:squarederror", eval_metric="rmse"
    )

def bootstrap_pred_interval(residuals: np.ndarray, yhat: float, B:int=200, alpha:float=0.05):
    """Intervalo por bootstrap de residuales (en escala original del target)."""
    if len(residuals) == 0:
        return yhat, yhat, yhat
    draws = np.random.choice(residuals, size=B, replace=True)
    dist = yhat + draws
    lo = float(np.quantile(dist, alpha/2))
    hi = float(np.quantile(dist, 1 - alpha/2))
    return yhat, lo, hi

# ===================== Vista rápida del histórico =====================
st.subheader("Histórico cargado")
st.dataframe(df.head(50), use_container_width=True)

# ===================== Entrenamiento y predicción =====================
metrics_rows = []
param_cache = {}
all_preds = []

for (g_edo, g_adm, g_suc), g0 in df.groupby(["edo", "adm", "sucursal"], dropna=False):
    g0 = g0.sort_values("fecha").reset_index(drop=True)

    if len(g0) < max(60, int(test_size_days) + 30):
        st.warning(f"Grupo ({g_edo}, {g_adm}, {g_suc}) con pocos datos (n={len(g0)}). Se omite.")
        continue

    # Enriquecer features
    g, FEATS = enrich_features(g0)

    # Transformación del target (log1p + shift para manejar ceros/negativos)
    y_min = float(g[TARGET].min())
    shift = -y_min + 1.0 if y_min <= 0 else 0.0
    g["_y_trans"] = np.log1p(g[TARGET] + shift)

    # Corte para validación simple
    cutoff = len(g) - int(test_size_days)
    train_df = g.iloc[:cutoff].copy()
    valid_df = g.iloc[cutoff:].copy()

    Xtr = train_df[FEATS]; ytr = train_df["_y_trans"]
    Xva = valid_df[FEATS]; yva = valid_df["_y_trans"]

    # Tuning por ADM (cache 1 vez por adm)
    if do_tune and g_adm not in param_cache:
        with st.spinner(f"Tuning ADM={g_adm} ..."):
            best_params = random_tune(Xtr, ytr, Xva, yva,
                                      n_samples=int(n_param_samples),
                                      early_stopping=int(early_stopping))
            param_cache[g_adm] = best_params
    params = param_cache.get(g_adm, dict(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.0, reg_lambda=1.0,
        random_state=42, n_jobs=4, objective="reg:squarederror", eval_metric="rmse"
    ))

    # Entrenamiento final (en espacio transformado)
    model = xgb.XGBRegressor(**params)
    safe_fit(model, Xtr, ytr, Xva=Xva, yva=yva, early_stopping=int(early_stopping))

    # Métrica simple en valid (transformado)
    yva_pred = model.predict(Xva)
    rmse_val = float(np.sqrt(mean_squared_error(yva, yva_pred)))

    # Residuales para bootstrap (volver a espacio original para intervalo)
    # Convertimos yva a original y pred a original:
    yva_orig = np.expm1(yva) - shift
    yva_pred_orig = np.expm1(yva_pred) - shift
    resid_val = (yva_orig - yva_pred_orig)

    # Backtest rolling (en espacio transformado)
    bt_input = g[FEATS + ["_y_trans"]].rename(columns={"_y_trans": "target_bt"})
    backtest_metrics = rolling_backtest(
        bt_input,
        features=FEATS, target="target_bt",
        n_folds=int(n_rolling_folds),
        params=params, early_stopping=int(early_stopping)
    )

    metrics_rows.append({
        "edo": g_edo, "adm": g_adm, "sucursal": g_suc,
        "rmse_val_log": rmse_val,
        "bt_folds": backtest_metrics["folds"],
        "bt_rmse_mean": backtest_metrics["rmse_mean"],
        "bt_rmse_std": backtest_metrics["rmse_std"],
        "bt_mape_mean": backtest_metrics["mape_mean"],
        "bt_mape_std": backtest_metrics["mape_std"],
    })

    # Tablas estacionales para futuras exógenas
    e_tab, s_tab, e_global, s_global = compute_estacional_tables(g0)

    # Generar futuro
    last_date = g0['fecha'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, int(n_days)+1)]
    s_flujo = list(g0[TARGET].values.astype(float))

    preds_rows = []

    def moving_avgs(series):
        s = pd.Series(series)
        mv3 = float(s.tail(3).mean()) if len(s) >= 1 else 0.0
        mv5 = float(s.tail(5).mean()) if len(s) >= 1 else 0.0
        mv10 = float(s.tail(10).mean()) if len(s) >= 1 else 0.0
        mv14 = float(s.tail(14).mean()) if len(s) >= 1 else 0.0
        return mv3, mv5, mv10, mv14

    for d in future_dates:
        cal = build_future_calendar(d)
        dow, month = cal["dia_semana"], cal["mes"]

        # Estimación estacional de entradas/salidas + ajuste de escenario
        e_future, s_future = get_future_es(dow, month, e_tab, s_tab, e_global, s_global, escenario_e, escenario_s)

        # Lags/medias sobre flujo
        lag1 = float(s_flujo[-1]) if len(s_flujo) >= 1 else 0.0
        lag2 = float(s_flujo[-2]) if len(s_flujo) >= 2 else lag1
        lag3 = float(s_flujo[-3]) if len(s_flujo) >= 3 else lag2
        lag5 = float(s_flujo[-5]) if len(s_flujo) >= 5 else lag3
        mv3, mv5, mv10, mv14 = moving_avgs(s_flujo)

        xrow = {
            "entradas": e_future,
            "salidas": s_future,
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
        xdf = pd.DataFrame([xrow])
        xdf, feats_all = enrich_features(xdf)
        xdf = xdf[feats_all]

        # Predicción en espacio transformado y regreso a original
        yhat_log = float(model.predict(xdf)[0])
        yhat = float(np.expm1(yhat_log) - shift)

        # Intervalo por bootstrap de residuales (en original)
        yhat_, lo, hi = bootstrap_pred_interval(residuals=resid_val.values, yhat=yhat, B=200, alpha=0.05)

        preds_rows.append({
            "edo": g_edo,
            "adm": g_adm,
            "sucursal": g_suc,
            "fecha_predicha": d,
            "yhat": yhat_,
            "yhat_lower": lo,
            "yhat_upper": hi,
            "modelo_version": f"xgb_v2_adm_{g_adm}",
            "trained_until": last_date,
            "features": json.dumps({**xrow, "shift": shift, "params": params})
        })

        s_flujo.append(yhat_)

    pred_df = pd.DataFrame(preds_rows)
    all_preds.append(pred_df)

# ===================== Resultados =====================
st.subheader("Resultados de predicción por grupo")
if all_preds:
    for pred_df in all_preds:
        g_edo = pred_df["edo"].iloc[0]
        g_adm = pred_df["adm"].iloc[0]
        g_suc = pred_df["sucursal"].iloc[0]
        st.markdown(f"**Grupo:** edo={g_edo} • adm={g_adm} • sucursal={g_suc}")
        st.dataframe(pred_df[["fecha_predicha","yhat","yhat_lower","yhat_upper"]], use_container_width=True)
else:
    st.warning("No se generaron predicciones (verifica filtros y tamaño de series).")

st.subheader("Métricas (valid simple + backtest rolling)")
if metrics_rows:
    metdf = pd.DataFrame(metrics_rows)
    st.dataframe(metdf, use_container_width=True)
else:
    st.info("Aún no hay métricas calculadas.")

# ===================== Guardado =====================
all_preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
if not all_preds_df.empty and st.button("Guardar predicciones en PostgreSQL (UPSERT)"):
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
    st.download_button("Descargar predicciones (CSV)", data=csv, file_name="predicciones_flujo.csv", mime="text/csv")

