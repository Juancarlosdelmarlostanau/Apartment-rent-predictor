"""Funciones de feature engineering, evaluación y predicción del modelo de precio de alquiler por m².

Reproducen los pasos del notebook modeling.ipynb.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score


TARGET = "price_per_m2"

COLUMNS_NOT_FEATURES = ["price", "price_per_m2", "cityname", "state"]

FEATURES_TOP = ["area_m2", "bedrooms", "latitude", "longitude", "dogs_allowed"]


def split_features_target(df):
    """Separa el dataset limpio en features y target, excluyendo price (data leakage) y las columnas de texto."""
    features = df.drop(columns=COLUMNS_NOT_FEATURES)
    target = df[TARGET]
    return features, target


def get_state_dummies(df):
    """Codifica la columna state con One-Hot Encoding (una columna binaria por estado)."""
    return pd.get_dummies(df["state"], prefix="state", dtype=int)


def build_final_features(X, state_dummies):
    """Construye la configuración final de 57 variables, en el orden que espera el modelo:
    5 variables principales + 51 estados + amenities_count."""
    return pd.concat([X[FEATURES_TOP], state_dummies.loc[X.index], X["amenities_count"]], axis=1)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Entrena el modelo y devuelve R², MAE y RMSE sobre el test."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return round(r2, 3), round(mae, 2), round(float(rmse), 2)


def evaluate_cv(model, X, y, cv=5):
    """Evalúa el modelo con validación cruzada y devuelve la media y la desviación típica del R²."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    return round(float(scores.mean()), 3), round(float(scores.std()), 3)


def save_artifacts(model, model_columns, model_path, columns_path):
    """Guarda el modelo (comprimido) y la lista de columnas en el orden que espera el modelo."""
    joblib.dump(model, model_path, compress=3)
    joblib.dump(list(model_columns), columns_path)


def load_artifacts(model_path, columns_path):
    """Carga el modelo y la lista de columnas guardados."""
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    return model, model_columns


def predict_price_per_m2(model, model_columns, cities, city, state, area_m2, bedrooms,
                         dogs_allowed, amenities_count):
    """Predice el precio por m² de una vivienda nueva.
    Las coordenadas se obtienen de la tabla de ciudades a partir de la ciudad y el estado."""
    location = cities[(cities["cityname"] == city) & (cities["state"] == state)].iloc[0]

    apartment = pd.DataFrame(0, index=[0], columns=model_columns)
    apartment["area_m2"] = area_m2
    apartment["bedrooms"] = bedrooms
    apartment["latitude"] = location["latitude"]
    apartment["longitude"] = location["longitude"]
    apartment["dogs_allowed"] = dogs_allowed
    apartment["amenities_count"] = amenities_count
    apartment[f"state_{state}"] = 1

    return float(model.predict(apartment)[0])