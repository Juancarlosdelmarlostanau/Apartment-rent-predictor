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
- Eliminación de variables sin valor predictivo (identificadores, columnas con un único valor, fecha de publicación).
- Los nulos de `amenities` y `pets_allowed` se interpretan como "no indicado" en lugar de eliminarse, conservando más de 3.500 anuncios.
- Transformación de servicios en variables individuales y de mascotas en variables binarias.
- Conversión de la superficie de pies² a m² y cálculo del target: **precio por m²**.
- Tratamiento de outliers con criterios justificados:
  - Superficie mínima de **20 m²**: por debajo, los casos corresponden a habitaciones sueltas o errores.
  - Precio mínimo de **5 $/m²**: por debajo predominan precios por habitación (por ejemplo, viviendas de estudiantes en ciudades universitarias).
- Resultado: **9.848 anuncios** (98,5% del dataset original) sin valores nulos.

### 3. Feature Engineering
- **Escalado:** comparación de `MinMaxScaler` y `StandardScaler`.
- **Selección de variables:** análisis de correlación con el target y entre variables. Se identifican 5 variables principales: `area_m2`, `bedrooms`, `latitude`, `longitude` y `dogs_allowed`.
- **Experimentos:** combinación sistemática de escalado (sí/no) y conjuntos de variables (todas / 5 principales / 4 descartadas).
- **Codificación de la ubicación:** One-Hot Encoding de `state` (51 estados).

### 4. Modelado
- Modelos comparados: KNN, regresión lineal y árbol de decisión.
- Métricas: R², MAE y RMSE.

##  Resultados

| Modelo | Configuración | R² | MAE ($/m²) | RMSE ($/m²) |
|---|---|---|---|---|
| KNN (baseline) | Sin escalar, todas las variables | 0,638 | 4,24 | 6,46 |
| KNN | StandardScaler, 5 variables principales | 0,622 | 4,21 | 6,60 |
| Regresión lineal | Todas las variables + estados | 0,544 | 4,75 | 7,26 |
| Árbol de decisión | Todas las variables | 0,665 | 3,90 | 6,22 |
| **Árbol de decisión** | **Todas las variables + estados** | **0,668** | **3,93** | **6,19** |

El árbol de decisión obtiene los mejores resultados. Las diferencias entre configuraciones son pequeñas y dependen de la partición train/test, por lo que se confirmarán con validación cruzada.

##  Hallazgos principales
- **La ubicación es clave:** viviendas de tamaño similar varían de 3.000 $/mes (Saint George, UT) a 19.500 $/mes (Los Ángeles, CA).
- **Escalado y selección de variables solo funcionan combinados en KNN:** escalar todas las variables empeora el modelo (los servicios añaden ruido), mientras que con las 5 variables principales se mantiene el rendimiento usando 5 variables en lugar de 36.
- **Las 4 variables descartadas no aportan información:** en los tres modelos obtienen un R² cercano a 0 o negativo.
- **La regresión lineal necesita la ubicación codificada:** con los estados, su R² pasa de 0,23 a 0,54, ya que no puede aprovechar las coordenadas directamente.
- **Precios por habitación:** en ciudades universitarias aparecen pisos grandes con alquileres muy bajos, publicados con el precio por habitación en lugar de por vivienda.

##  Limitaciones

- Los datos son de 2019 (previos a la pandemia), por lo que el modelo refleja precios de ese momento.
- Algunos estados tienen pocos anuncios, lo que reduce la fiabilidad del modelo en esas zonas.
- Las viviendas de gama alta son escasas y el modelo comete errores mayores con ellas.
- Los resultados actuales se basan en una única partición train/test, lo que genera cierta variabilidad.

##  Próximos pasos

- Validación cruzada y ajuste de hiperparámetros con Grid Search.
- Pruebas con modelos ensemble.
- Modularización del código en scripts de Python.
- Explorar la creación de un detector de oportunidades a partir de las predicciones del modelo.
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
│   └── 02_modeling.ipynb                            # Feature engineering y Modelado
├── scrips/                                          # Scripts de Python
├── app/                                             # Aplicación de Streamlit
└── README.md

##  Autor

**Juan Carlos Del Mar Lostanau**
Arquitecto/student ironhack 