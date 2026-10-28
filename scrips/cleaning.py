"""Funciones de limpieza del dataset Apartment for Rent Classified (UCI).

Reproducen, en el mismo orden, los pasos del notebook data_exploration.ipynb.
"""

import pandas as pd


SQFT_TO_M2 = 0.092903

COLUMNS_TO_DROP = ["id", "title", "body", "price_display", "address", "currency",
                   "fee", "source", "category", "price_type", "time"]

KEY_COLUMNS = ["bathrooms", "bedrooms", "latitude", "longitude", "cityname", "state"]


def load_raw_data(path):
    """Carga el dataset original de UCI (separado por ';')."""
    return pd.read_csv(path, sep=";", encoding="cp1252")


def filter_listings(df):
    """Conserva solo apartamentos con precio mensual y elimina los dos errores evidentes
    (estudio de 52.500 $ en Barstow y vivienda de 40.000 pies² en Nueva York)."""
    df = df[(df["category"] == "housing/rent/apartment") & (df["price_type"] == "Monthly")]
    df = df[(df["price"] < 50000) & (df["square_feet"] < 40000)]
    return df


def drop_unused_columns(df):
    """Elimina las columnas sin valor predictivo o con data leakage."""
    return df.drop(columns=COLUMNS_TO_DROP)


def drop_missing_values(df):
    """Elimina las filas con nulos en las columnas clave (menos del 1% de los datos)."""
    return df.dropna(subset=KEY_COLUMNS)


def encode_pets(df):
    """Convierte pets_allowed en dos columnas binarias. Los nulos se interpretan como 'no indicado' (0)."""
    df = df.copy()
    df["cats_allowed"] = df["pets_allowed"].str.contains("Cats", na=False).astype(int)
    df["dogs_allowed"] = df["pets_allowed"].str.contains("Dogs", na=False).astype(int)
    return df.drop(columns="pets_allowed")


def encode_amenities(df):
    """Crea una columna binaria por servicio y amenities_count con el total de servicios.
    Los nulos se interpretan como 'sin servicios' y se elimina la columna vacía que generan las comas sobrantes."""
    df = df.copy()
    amenities_dummies = df["amenities"].fillna("").str.get_dummies(sep=",")
    amenities_dummies = amenities_dummies.drop(columns="", errors="ignore")
    df = pd.concat([df, amenities_dummies], axis=1)
    df["amenities_count"] = amenities_dummies.sum(axis=1)
    return df.drop(columns="amenities")


def encode_photo(df):
    """Convierte has_photo en binaria: 1 si el anuncio tiene foto (normal o miniatura), 0 si no."""
    df = df.copy()
    df["has_photo"] = (df["has_photo"] != "No").astype(int)
    return df


def add_target(df):
    """Convierte la superficie a m² y calcula el target: precio por m²."""
    df = df.copy()
    df["area_m2"] = df["square_feet"] * SQFT_TO_M2
    df["price_per_m2"] = df["price"] / df["area_m2"]
    return df.drop(columns="square_feet")


def remove_outliers(df, min_area=20, min_price_m2=5):
    """Elimina viviendas de menos de 20 m² (habitaciones o errores) y precios
    inferiores a 5 $/m² (precios por habitación)."""
    df = df[df["area_m2"] >= min_area]
    df = df[df["price_per_m2"] >= min_price_m2]
    return df


def clean_data(df):
    """Aplica todo el proceso de limpieza, en orden, al dataset original."""
    df = filter_listings(df)
    df = drop_unused_columns(df)
    df = drop_missing_values(df)
    df = encode_pets(df)
    df = encode_amenities(df)
    df = encode_photo(df)
    df = add_target(df)
    df = remove_outliers(df)
    return df