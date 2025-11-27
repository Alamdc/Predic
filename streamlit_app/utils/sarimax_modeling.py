import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_squared_error
import warnings

# Suprimimos warnings para limpiar la UI
warnings.filterwarnings("ignore")

def optimize_sarimax(series, exog=None, seasonal_period=7):
    """
    Utiliza Auto-ARIMA (Stepwise search) para encontrar el mejor modelo
    de forma eficiente y estadísticamente robusta.
    
    Args:
        series: La serie temporal objetivo (Pandas Series).
        exog: (Opcional) DataFrame o Matriz de variables exógenas.
        seasonal_period: Periodo estacional (m). 7 para diario/semanal.
    
    Returns:
        model_fit: El objeto ARIMA entrenado.
        order: Tupla (p,d,q).
        seasonal_order: Tupla (P,D,Q,s).
        aic: Métrica de calidad.
    """
    
    # Auto-ARIMA se encarga de buscar los mejores p,d,q,P,D,Q
    # stepwise=True hace que sea mucho más rápido que un GridSearch completo
    model = pm.auto_arima(
        y=series,
        X=exog,                # Inyectamos variables exógenas si existen
        start_p=0, start_q=0,
        max_p=3, max_q=3,      # Ampliamos el rango de búsqueda
        m=seasonal_period,     # Estacionalidad (ej. 7 días)
        start_P=0, seasonal=True,
        d=None,                # 'None' permite que el algoritmo calcule 'd' óptimo (test ADF)
        D=None,                # 'None' permite calcular 'D' óptimo
        trace=False,           # Poner True si quieres ver el log en consola
        error_action='ignore',  
        suppress_warnings=True, 
        stepwise=True          # Algoritmo inteligente (no fuerza bruta)
    )

    # Extraer parámetros
    order = model.order
    seasonal_order = model.seasonal_order
    aic = model.aic()

    return model, order, seasonal_order, aic

def generate_sarimax_forecast(model, steps=7, exog_future=None, alpha=0.05):
    """
    Genera predicciones futuras usando el modelo pmdarima.
    
    Args:
        model: El modelo entrenado por pmdarima.
        steps: Días a futuro.
        exog_future: (Opcional) Variables exógenas futuras. 
                     ¡OJO! Deben conocerse a futuro (ej. día de la semana, feriado).
    """
    # Generar predicción e intervalos
    # pmdarima devuelve (predicciones, intervalos_confianza)
    preds, conf_int = model.predict(
        n_periods=steps, 
        X=exog_future, 
        return_conf_int=True, 
        alpha=alpha
    )
    
    # Formatear salida para que coincida con tu código anterior
    # conf_int viene como matriz numpy [[low, high], ...]
    conf_df = pd.DataFrame(conf_int, columns=["lower", "upper"])
    
    # preds es una Serie o array, aseguramos consistencia
    if isinstance(preds, pd.Series):
        preds = preds.values
        
    return preds, conf_df