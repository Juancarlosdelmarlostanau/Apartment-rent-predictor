#  Apartment Rent Predictor

Predicción del precio de alquiler por superficie(m2) de viviendas en Estados Unidos mediante Machine Learning.

##  Objetivo

Desarrollar un modelo de Machine Learning capaz de estimar el precio de alquiler por m² de una vivienda a partir de sus características (superficie, habitaciones, baños, servicios, ubicación, etc.).

- Explorar la creación de un detector de oportunidades a partir de las predicciones del modelo.

##  Datos

- **Fuente:** [Apartment for Rent Classified – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified)
- **Descripción:** anuncios de viviendas en alquiler en Estados Unidos, publicados entre septiembre y diciembre de 2019.
- **Versión utilizada:** 10.000 registros y 22 variables.
- **Licencia:** CC BY 4.0

##  Metodología

### 1. Exploración de datos
- Análisis de tipos de variables, valores nulos y distribuciones.
- Detección de outliers mediante boxplots y revisión de casos extremos con contexto (ubicación, superficie, habitaciones).
- Identificación de variables con *data leakage* ('title', 'body', 'price_display'), que contienen el precio en texto.

### 2. Limpieza de datos
- Filtrado de apartamentos con precio mensual.
- Eliminación de errores evidentes (un estudio de 52.500 $/mes y una vivienda de 40.000 pies² con un dormitorio).
- Eliminación de variables sin valor predictivo (identificadores, columnas con un único valor, fecha de publicación).
- Los nulos de 'amenities' y 'pets_allowed' se interpretan como "no indicado" en lugar de eliminarse, conservando más de 3.500 anuncios.
- Transformación de servicios en variables individuales y de mascotas en variables binarias.
- Conversión de la superficie de pies² a m² y cálculo del target: **precio por m²**.
- Resultado: **9.878 anuncios** (98,8% del dataset original) sin valores nulos.

### 3. Modelado
- Modelo base (*baseline*): KNN Regressor con parámetros por defecto.

##  Resultados

Modelo : KNN baseline (k=5, sin escalar) = 0.59
MAE ($/m²)  = 4,36 
RMSE ($/m²) = 6,74

El modelo base explica el 59% de la variación del precio por m², con un error medio de 4,36 $/m². Servirá como referencia para medir las mejoras.

##  Hallazgos principales

- **La ubicación es clave:** viviendas de tamaño similar varían de 3.000 $/mes (Saint George, UT) a 19.500 $/mes (Los Ángeles, CA).
- **Distribución geográfica desequilibrada:** Texas y California concentran más de una cuarta parte de los anuncios.

##  Limitaciones

- Los datos son de 2019 (previos a la pandemia), por lo que el modelo refleja precios de ese momento.
- Algunos estados tienen pocos anuncios, lo que reduce la fiabilidad del modelo en esas zonas.

##  Próximos pasos

- Feature engineering: codificación de la ubicación y revisión de outliers del precio por m².
- Ajuste de hiperparámetros con Grid Search y Cross-Validation.
- Pruebas con modelos ensemble.
- Aplicación interactiva con Streamlit.

##  Estructura del proyecto

apartment-rent-predictor/
├── data/
│   ├── raw/
│   │   └── apartments_for_rent_classified_10K.csv   # Dataset original
│   └── clean/
│       └── apartments_clean.csv                     # Dataset limpio
├── notebooks/
│   ├── 01_data_exploration.ipynb                    # Exploración y limpieza
│   └── 02_modeling.ipynb                            # Modelado
├── scrips/                                          # Scripts de Python
├── app/                                             # Aplicación de Streamlit
└── README.md

##  Autor

**Juan Carlos Del Mar Lostanau**
Arquitecto/student ironhack 