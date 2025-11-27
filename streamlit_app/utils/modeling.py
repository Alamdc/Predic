import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterSampler
from datetime import date

# Variables constantes
BASE_FEATURES = [
    "entradas", "salidas",
    "anio", "trimestre", "mes", "semana_anio", "dia_semana",
    "media_movil_3", "media_movil_5", "media_movil_10", "media_movil_14",
    "lag_1", "lag_2", "lag_3", "lag_5"
]
TARGET = "flujo_efectivo"

def safe_fit(model, Xtr, ytr, Xva=None, yva=None, early_stopping=0):
    """Compatible con distintas versiones de XGBoost."""
    fit_kwargs = {}
    if Xva is not None and yva is not None:
        fit_kwargs["eval_set"] = [(Xva, yva)]

    if early_stopping and Xva is not None:
        # Intento 1: Callbacks
        try:
            cb = xgb.callback.EarlyStopping(rounds=int(early_stopping), min_delta=0.0, save_best=True)
            model.fit(Xtr, ytr, verbose=False, callbacks=[cb], **fit_kwargs)
            return model
        except Exception:
            pass
        # Intento 2: early_stopping_rounds parameter
        try:
            model.fit(Xtr, ytr, verbose=False, early_stopping_rounds=int(early_stopping), **fit_kwargs)
            return model
        except Exception:
            pass

    try:
        model.fit(Xtr, ytr, verbose=False, **fit_kwargs)
    except Exception:
        model.fit(Xtr, ytr, **fit_kwargs)
    return model

def enrich_features(g: pd.DataFrame):
    g = g.copy()
    
    # --- 1. PREPARACIÓN BÁSICA ---
    if 'fecha' in g.columns:
        if not np.issubdtype(g['fecha'].dtype, np.datetime64):
            g['fecha'] = pd.to_datetime(g['fecha'])
        g = g.sort_values('fecha')
        
        # --- 2. CARACTERÍSTICAS DE CALENDARIO ---
        g['dia'] = g['fecha'].dt.day
        g['mes'] = g['fecha'].dt.month
        g['dia_semana'] = g['fecha'].dt.dayofweek
        
        g['es_fin_de_semana'] = g['dia_semana'].isin([5, 6]).astype(int)
        g['es_quincena'] = g['dia'].isin([15, 30, 31]).astype(int)
        g['es_inicio_mes'] = (g['dia'] <= 5).astype(int)
        g['es_fin_mes'] = g['fecha'].dt.is_month_end.astype(int)

        g['sin_dia_semana'] = np.sin(2 * np.pi * g['dia_semana'] / 7)
        g['cos_dia_semana'] = np.cos(2 * np.pi * g['dia_semana'] / 7)
        g['sin_mes'] = np.sin(2 * np.pi * g['mes'] / 12)
        g['cos_mes'] = np.cos(2 * np.pi * g['mes'] / 12)
    
    # --- 3. LAGS (PROTEGIDO) ---
    target = 'flujo_efectivo'
    
    # SOLO calculamos lags si tenemos la historia (Entrenamiento)
    if target in g.columns:
        g['lag_1'] = g[target].shift(1)
        g['lag_7'] = g[target].shift(7)
        g['lag_14'] = g[target].shift(14)
        g['lag_30'] = g[target].shift(30)
        
        g['media_movil_7'] = g[target].rolling(window=7).mean().shift(1)
        g['media_movil_30'] = g[target].rolling(window=30).mean().shift(1)
        g['std_movil_7'] = g[target].rolling(window=7).std().shift(1)
    
    # --- 4. DEFINICIÓN DE FEATURES ---
    features = [
        'dia', 'mes', 'dia_semana', 
        'es_fin_de_semana', 'es_quincena', 'es_inicio_mes', 'es_fin_mes',
        'sin_dia_semana', 'cos_dia_semana', 'sin_mes', 'cos_mes',
        'lag_1', 'lag_7', 'lag_14', 'lag_30',
        'media_movil_7', 'media_movil_30', 'std_movil_7'
    ]
    
    # Rellenar NaNs con 0 para evitar errores en predicción si faltan datos
    # Esto asegura que si 'xrow' envía los lags, se respeten, y si faltan, sean 0.
    for col in features:
        if col not in g.columns:
            g[col] = 0.0
            
    g = g.fillna(0)
    
    return g, features

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
    g = g.copy()
    g["dia_sem"] = g["dia_semana"]
    g["m"] = g["mes"]
    e_tab = g.pivot_table(index="dia_sem", columns="m", values="entradas", aggfunc="median")
    s_tab = g.pivot_table(index="dia_sem", columns="m", values="salidas", aggfunc="median")
    e_global = float(np.nanmedian(g["entradas"])) if "entradas" in g else 0.0
    s_global = float(np.nanmedian(g["salidas"])) if "salidas" in g else 0.0
    return e_tab, s_tab, e_global, s_global

def get_future_es(dia_sem:int, month:int, e_tab, s_tab, e_global:float, s_global:float, adj_e:float, adj_s:float):
    e = e_tab.loc[dia_sem, month] if (dia_sem in e_tab.index and month in e_tab.columns) else np.nan
    s = s_tab.loc[dia_sem, month] if (dia_sem in s_tab.index and month in s_tab.columns) else np.nan
    e = float(e) if not np.isnan(e) else e_global
    s = float(s) if not np.isnan(s) else s_global
    e = e * (1.0 + adj_e/100.0)
    s = s * (1.0 + adj_s/100.0)
    return e, s

def rolling_backtest(series_df: pd.DataFrame, features: list, target: str, n_folds:int=3, params:dict=None, early_stopping:int=50):
    if params is None:
        params = dict(n_estimators=400, max_depth=6, learning_rate=0.05, objective="reg:squarederror")
    n = len(series_df)
    fold_size = max(7, n // (n_folds + 1))
    rmses, mapes = [], []

    for k in range(1, n_folds+1):
        split = k * fold_size
        if split >= n-7: break
        train = series_df.iloc[:split]
        test = series_df.iloc[split:split+fold_size]
        if len(test) < 7: continue

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
        "rmse_prom": float(np.mean(rmses)) if rmses else None,
        "rmse_desv": float(np.std(rmses)) if rmses else None,
        "mape_prom": float(np.mean(mapes)) if mapes else None,
        "mape_desv": float(np.std(mapes)) if mapes else None,
    }

def random_tune(Xtr, ytr, Xte, yte, n_samples:int=20, early_stopping:int=50, seed:int=42):
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
        params = dict(objective="reg:squarederror", n_jobs=4, random_state=42, eval_metric="rmse", **p)
        model = xgb.XGBRegressor(**params)
        safe_fit(model, Xtr, ytr, Xva=Xte, yva=yte, early_stopping=early_stopping)
        pred = model.predict(Xte)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best = params

    return best if best is not None else dict(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror"
    )

def bootstrap_pred_interval(residuals: np.ndarray, yhat: float, B:int=200, alpha:float=0.05):
    if len(residuals) == 0: return yhat, yhat, yhat
    draws = np.random.choice(residuals, size=B, replace=True)
    dist = yhat + draws
    lo = float(np.quantile(dist, alpha/2))
    hi = float(np.quantile(dist, 1 - alpha/2))
    return yhat, lo, hi