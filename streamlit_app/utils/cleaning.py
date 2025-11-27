import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def limpiar_anomalias_isolation_forest(df, columna='flujo_efectivo', contaminacion=0.02):
    """
    Detecta outliers usando Isolation Forest y los reemplaza por interpolación.
    Args:
        df: DataFrame con los datos.
        columna: Nombre de la columna a limpiar.
        contaminacion: Porcentaje estimado de outliers (0.02 = 2% de datos sucios).
    """
    data = df.copy()
    
    # 1. Preparar datos (Isolation Forest necesita 2D array)
    X = data[[columna]].values
    
    # 2. Entrenar Isolation Forest
    # n_estimators=100 es estándar, random_state para reproducibilidad
    iso = IsolationForest(contamination=contaminacion, random_state=42)
    preds = iso.fit_predict(X)
    
    # preds: 1 = normal, -1 = anómalo
    data['is_outlier'] = preds
    
    # 3. Contar cuántos detectó
    n_outliers = (data['is_outlier'] == -1).sum()
    print(f"Isolation Forest detectó y limpió {n_outliers} anomalias.")
    
    # 4. IMPUTACIÓN (Clave para series de tiempo)
    # Convertimos los outliers a NaN para luego interpolarlos
    data.loc[data['is_outlier'] == -1, columna] = np.nan
    
    # Interpolación lineal (rellena el hueco conectando el punto anterior y el siguiente)
    # limit_direction='both' ayuda si el outlier es el primer o último dato
    data[columna] = data[columna].interpolate(method='linear', limit_direction='both')
    
    return data.drop(columns=['is_outlier'])