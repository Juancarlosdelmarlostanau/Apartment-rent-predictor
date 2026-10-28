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

## Metodología

### 1. Exploración de datos
- Análisis de tipos de variables, valores nulos y distribuciones.
- Detección de outliers mediante boxplots y revisión de casos extremos con contexto (ubicación, superficie, dormitorios).
- Identificación de variables con *data leakage* (`title`, `body`, `price_display`), que contienen el precio en texto.

### 2. Limpieza de datos
- Filtrado de apartamentos con precio mensual y eliminación de errores evidentes.
- Eliminación de variables sin valor predictivo (identificadores, columnas con un único valor, fecha de publicación).
- Los nulos de `amenities` y `pets_allowed` se interpretan como "no indicado", conservando más de 3.500 anuncios.
- Transformación de servicios en variables individuales y de mascotas en variables binarias.
- Conversión de la superficie a m² y cálculo del target: **precio por m²**.
- Tratamiento de outliers con criterios justificados: superficie mínima de **20 m²** (habitaciones o errores) y precio mínimo de **5 $/m²** (precios por habitación, frecuentes en ciudades universitarias).
- Resultado: **9.848 anuncios** (98,5% del dataset original) sin valores nulos.

### 3. Feature Engineering
- **Escalado:** comparación de `MinMaxScaler` y `StandardScaler`.
- **Selección de variables:** correlación con el target y entre variables. Se identifican 5 variables principales: `area_m2`, `bedrooms`, `latitude`, `longitude` y `dogs_allowed`.
- **Codificación de la ubicación:** One-Hot Encoding de `state` (51 estados).
- **Interpretación:** coeficientes de la regresión lineal e importancia de variables del árbol de decisión.

### 4. Modelado
- **Modelos base:** KNN, regresión lineal y árbol de decisión.
- **Modelos ensemble:** Bagging, Pasting, Random Forest, AdaBoost y Gradient Boosting, comparados en siete configuraciones de variables.
- **Validación cruzada (5-fold)** de los finalistas (Random Forest y AdaBoost).
- **Ajuste de hiperparámetros:** Grid Search (125 combinaciones) y Random Search (200 combinaciones sobre más de 67.000 posibles).
- **Métricas:** R², MAE y RMSE.

### 5. Modelo final
**Random Forest** con 57 variables (5 principales + `amenities_count` + estados), ajustado con Random Search: `n_estimators=400`, `max_depth=25`, `max_leaf_nodes=2866`, `max_features=0.6`, `min_samples_leaf=2`.

## Resultados

| Etapa | Modelo | Variables | R² | MAE ($/m²) | RMSE ($/m²) |
|---|---|---|---|---|---|
| Baseline | KNN | Todas (36) | 0,638 | 4,24 | 6,46 |
| Feature engineering | KNN + StandardScaler | 5 principales | 0,622 | 4,21 | 6,60 |
| Otros modelos | Árbol de decisión | 5 principales + estados | 0,653 | 3,94 | 6,33 |
| Ensemble | Random Forest | 57 | 0,787 | 3,10 | 4,96 |
| **Modelo final** | **Random Forest (Random Search)** | **57** | **0,794** | **3,03** | **4,87** |

Del baseline al modelo final, el R² mejora en 0,16 puntos y el error medio se reduce un 29%. En un piso típico de 75 m², el error medio es de unos 227 $/mes. El resultado en el test es coherente con la validación cruzada (R² = 0,788), lo que confirma que el modelo generaliza bien.

## Hallazgos principales

- **La ubicación es lo más importante:** longitud, latitud y estado aportan más del 60% de la importancia del modelo.
- **California concentra el efecto de los estados:** por sí sola recoge casi todo el efecto geográfico de los 51 estados.
- **La correlación no lo cuenta todo:** `amenities_count`, descartada por su baja correlación, mejora los modelos ensemble, que aprovechan combinaciones de variables.
- **Los modelos ensemble aportan la mayor mejora** (R² de 0,65 a 0,79). El ajuste de hiperparámetros añade una mejora pequeña: Random Forest es robusto y estaba cerca de su techo.
- **`max_features` es el hiperparámetro clave:** limitar las variables que ve cada árbol (60%) mejora el modelo.
- **Precios por habitación:** en ciudades universitarias aparecen pisos grandes anunciados con el precio por habitación en lugar de por vivienda.

## Limitaciones

- Los datos son de 2019 (previos a la pandemia), por lo que el modelo refleja precios de ese momento.
- La distribución geográfica está desequilibrada: Texas y California concentran más de una cuarta parte de los anuncios.
- Las viviendas de gama alta son escasas y el modelo comete errores mayores con ellas.
- Grid Search y Random Search no garantizan los parámetros óptimos, sino los mejores de entre los probados.

## Aplicación

_(En construcción: aplicación interactiva con Streamlit y detector de oportunidades)_

## 📁 Estructura del proyecto

```
Apartment-rent-predictor/
├── data/
│   ├── raw/
│   │   └── apartments_for_rent_classified_10K.csv   # Dataset original
│   └── clean/
│       ├── apartments_clean.csv                     # Dataset limpio
│       └── cities.csv                               # Coordenadas medias y estado por ciudad
├── notebooks/
│   ├── data_exploration.ipynb                       # Exploración y limpieza
│   └── modeling.ipynb                               # Feature engineering, modelado y ajuste
├── scrips/
│   ├── cleaning.py                                  # Funciones de limpieza
│   └── modeling.py                                  # Funciones de feature engineering, evaluación y predicción
├── models/
│   ├── rent_model.pkl                               # Modelo final (Random Forest)
│   └── model_columns.pkl                            # Columnas que espera el modelo
├── app/                                             # Aplicación de Streamlit
└── README.md
```


##  Autor

**Juan Carlos Del Mar Lostanau**
Arquitecto/student ironhack 