import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from scrips.modeling import load_artifacts, predict_price_per_m2

MODEL_MAE = 3.03  # Error medio del modelo final en el test ($/m²)
MIN_LISTINGS = 5  # Por debajo de este número de anuncios, la estimación de la ciudad es menos fiable

STATE_NAMES = {
    "AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DC": "Washington D.C.", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "IA": "Iowa", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "MA": "Massachusetts", "MD": "Maryland", "ME": "Maine", "MI": "Michigan", "MN": "Minnesota",
    "MO": "Missouri", "MS": "Mississippi", "MT": "Montana", "NC": "North Carolina",
    "ND": "North Dakota", "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NV": "Nevada", "NY": "New York", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VA": "Virginia",
    "VT": "Vermont", "WA": "Washington", "WI": "Wisconsin", "WV": "West Virginia", "WY": "Wyoming",
}

st.set_page_config(page_title="Apartment Rent Predictor", page_icon="🏢")


@st.cache_resource
def load_model():
    """Carga el modelo y sus columnas una sola vez."""
    return load_artifacts(ROOT / "models" / "rent_model.pkl", ROOT / "models" / "model_columns.pkl")


@st.cache_data
def load_cities():
    """Carga la tabla de ciudades con sus coordenadas medias, su estado y su número de anuncios."""
    return pd.read_csv(ROOT / "data" / "clean" / "cities.csv")


def default_index(options, default):
    """Devuelve la posición del valor por defecto en la lista de opciones (o 0 si no está)."""
    return options.index(default) if default in options else 0


model, model_columns = load_model()
cities = load_cities()

# ---------- Cabecera ----------
st.title("🏢 Apartment Rent Predictor")
st.write("Estima el precio de alquiler de una vivienda en Estados Unidos y comprueba si un anuncio está por encima o por debajo del mercado.")
st.caption("Modelo: Random Forest entrenado con anuncios de 2019 · R² = 0,79 · Error medio = 3,03 $/m²")

# ---------- 1. Datos de la vivienda ----------
st.header("1. Datos de la vivienda")

states = sorted(cities["state"].unique())

col1, col2 = st.columns(2)
with col1:
    state = st.selectbox("Estado", states, index=default_index(states, "TX"),
                         format_func=lambda code: f"{STATE_NAMES.get(code, code)} ({code})")
    state_cities = sorted(cities.loc[cities["state"] == state, "cityname"])
    city = st.selectbox("Ciudad", state_cities, index=default_index(state_cities, "Austin"))
    area_m2 = st.number_input("Superficie (m²)", min_value=20, max_value=500, value=75)
with col2:
    bedrooms = st.number_input("Dormitorios", min_value=0, max_value=9, value=2, help="0 = estudio")
    amenities_count = st.slider("Número de servicios", min_value=0, max_value=27, value=5,
                                help="Parking, piscina, gimnasio, ascensor...")
    dogs_allowed = st.checkbox("Se admiten perros")

location = cities[(cities["cityname"] == city) & (cities["state"] == state)].iloc[0]

st.map(pd.DataFrame({"lat": [location["latitude"]], "lon": [location["longitude"]]}), zoom=9)

if location["listings"] < MIN_LISTINGS:
    st.warning(f"⚠️ Solo hay {location['listings']} anuncio(s) de {city} en los datos de entrenamiento: "
               "la estimación para esta ciudad es menos fiable.")

# ---------- 2. Precio estimado ----------
price_m2 = predict_price_per_m2(model, model_columns, cities, city, state,
                                area_m2, bedrooms, int(dogs_allowed), amenities_count)
monthly_rent = price_m2 * area_m2

st.header("2. Precio estimado")

col3, col4 = st.columns(2)
col3.metric("Precio por m²", f"{price_m2:.2f} $/m²")
col4.metric("Alquiler mensual", f"{monthly_rent:,.0f} $/mes")

# ---------- 3. Detector de oportunidades ----------
st.header("3. Detector de oportunidades")

listed_rent = st.number_input("Alquiler mensual del anuncio ($/mes)", min_value=0, value=0, step=50,
                              help="Introduce el precio de un anuncio real para compararlo con el estimado. Déjalo en 0 si no tienes ninguno.")

if listed_rent > 0:
    margin = MODEL_MAE * area_m2
    difference_pct = (listed_rent - monthly_rent) / monthly_rent * 100

    if listed_rent > monthly_rent + margin:
        st.error(f"🔴 **Sobrevalorado:** el anuncio está un {difference_pct:.0f}% por encima del precio estimado.")
    elif listed_rent < monthly_rent - margin:
        st.success(f"🟢 **Oportunidad:** el anuncio está un {abs(difference_pct):.0f}% por debajo del precio estimado.")
    else:
        st.info(f"🔵 **Precio de mercado:** la diferencia ({difference_pct:+.0f}%) está dentro del margen de error del modelo.")

    st.caption(f"Margen de tolerancia: ±{margin:,.0f} $/mes (error medio del modelo × superficie).")