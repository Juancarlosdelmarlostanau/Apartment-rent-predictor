# Apartment Rent Predictor

Predicción del precio de alquiler por m2, de viviendas en Estados Unidos mediante Machine Learning.
**🔗 Aplicación en línea:** [apartment-rent-predictor.streamlit.app](https://apartment-rent-predictor.streamlit.app)

##  Objetivo

- Desarrollar un modelo de Machine Learning capaz de estimar el precio de alquiler por m² de una vivienda a partir de sus características (superficie, dormitorios, servicios y ubicación)
- integrarlo en una aplicación interactiva con un detector de oportunidades que identifica anuncios sobrevalorados o por debajo del mercado.


##  Datos

- **Fuente:** [Apartment for Rent Classified – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified)
- **Descripción:** anuncios de viviendas en alquiler en Estados Unidos, publicados entre septiembre y diciembre de 2019.
- **Versión de entrenamiento:** 10.000 registros y 22 variables (incluida en `data/raw/`).
- **Versión de prueba:** versión completa de 100.000 registros, usada solo para evaluar el modelo con datos nuevos. No se incluye en el repositorio por su tamaño: puede descargarse desde la página de UCI y guardarse en `data/raw/` para ejecutar el notebook `new_data_test.ipynb`.
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
- **Ajuste de hiperparámetros:** Grid Search (125 combinaciones, 625 entrenamientos) y Random Search (200 combinaciones, 1.000 entrenamientos) aplicados a los dos finalistas, Random Forest y AdaBoost, con los mismos rangos de valores y validación cruzada de 5 particiones.
- **Métricas:** R², MAE y RMSE.

### 5. Modelo final
**Random Forest** con 57 variables (5 principales + `amenities_count` + estados), ajustado con Random Search: `n_estimators=400`, `max_depth=25`, `max_leaf_nodes=2866`, `max_features=0.6`, `min_samples_leaf=2`.

### 6. Prueba con datos nuevos
- Evaluación del modelo final, sin reentrenar, con la versión completa del dataset.
- Eliminación de los 9.983 anuncios que ya estaban en la versión de entrenamiento, para que la prueba sea independiente.
- Limpieza con el mismo pipeline (`clean_data()`): 88.913 anuncios nuevos.
- Análisis del error según el número de anuncios de cada ciudad en el entrenamiento.

## Resultados

| Etapa | Modelo | Variables | R² | MAE ($/m²) | RMSE ($/m²) |
|---|---|---|---|---|---|
| Baseline | KNN | Todas (36) | 0,638 | 4,24 | 6,46 |
| Feature engineering | KNN + StandardScaler | 5 principales | 0,622 | 4,21 | 6,60 |
| Otros modelos | Árbol de decisión | 5 principales + estados | 0,653 | 3,94 | 6,33 |
| Ensemble | Random Forest | 57 | 0,787 | 3,10 | 4,96 |
| **Modelo final** | **Random Forest (Random Search)** | **57** | **0,794** | **3,03** | **4,87** |

Del baseline al modelo final, el R² mejora en 0,16 puntos y el error medio se reduce un 29%. En un piso típico de 75 m², el error medio es de unos 227 $/mes. El resultado en el test es coherente con la validación cruzada (R² = 0,788), lo que confirma que el modelo generaliza bien.

### Prueba con datos nuevos

| Evaluación | Anuncios evaluados | R² | MAE ($/m²) | RMSE ($/m²) |
|---|---|---|---|---|
| Test (20% del dataset de 10.000) | 1.970 | 0,794 | 3,03 | 4,87 |
| Datos nuevos (dataset de 100.000) | 88.913 | 0,668 | 3,29 | 5,10 |

Con 88.913 anuncios que nunca ha visto, el error medio apenas aumenta (un 9%), lo que confirma que el modelo generaliza bien. El R² baja porque los datos nuevos incluyen muchas ciudades poco o nada representadas en el entrenamiento: en ciudades que el modelo no vio nunca, el R² es de 0,53, mientras que en ciudades con 5 o más anuncios de entrenamiento sube a 0,71.

## Hallazgos principales

- **La ubicación es lo más importante:** longitud, latitud y estado aportan más del 60% de la importancia del modelo.
- **California concentra el efecto de los estados:** por sí sola recoge casi todo el efecto geográfico de los 51 estados.
- **La correlación no lo cuenta todo:** `amenities_count`, descartada por su baja correlación, mejora los modelos ensemble, que aprovechan combinaciones de variables.
- **Los modelos ensemble aportan la mayor mejora** (R² de 0,65 a 0,79). El ajuste de hiperparámetros añade una mejora pequeña: Random Forest es robusto y estaba cerca de su techo.
- **`max_features` es el hiperparámetro clave:** limitar las variables que ve cada árbol (60%) mejora el modelo.
- **Los modelos ensemble aportan la mayor mejora:** pasar de un árbol de decisión a Random Forest eleva el R² de 0,65 a 0,79, el mayor salto de todo el proyecto.
- **El modelo ha alcanzado su techo con los datos disponibles:** Grid Search no mejora el resultado en el test (R² = 0,787 con los parámetros iniciales y con los ajustados), y Random Search, tras probar 200 combinaciones sobre más de 67.000 posibles, solo lo eleva a 0,794. Además, pasar de 30 a 200 combinaciones apenas aporta (+0,003). El rendimiento ya no depende de la configuración del modelo, sino de la información disponible: para mejorar de forma significativa harán falta más datos o nuevas variables.
- **El precio depende del barrio, no solo de la ciudad:** en los datos nuevos, las grandes ciudades presentan el mejor R² pero no el menor error, porque sus precios varían mucho más entre barrios. Las variables actuales no capturan esa escala urbana.

## Aplicación

Aplicación interactiva desarrollada con **Streamlit** que permite:

- **Estimar el alquiler** de una vivienda a partir de su ubicación (estado y ciudad), superficie, dormitorios, número de servicios y admisión de perros. Las coordenadas se obtienen automáticamente de la ciudad elegida.
- **Detectar oportunidades:** al introducir el precio de un anuncio real, la aplicación lo compara con la estimación y lo clasifica como 🔴 sobrevalorado, 🔵 precio de mercado o 🟢 oportunidad. El margen de tolerancia es el error medio del modelo (3,03 $/m²) multiplicado por la superficie.
- **Visualizar la ubicación** en un mapa y avisar cuando la ciudad tiene pocos anuncios (menos de 5) en los datos de entrenamiento.

### Cómo ejecutarla

La aplicación está publicada en **Streamlit Community Cloud** y puede usarse directamente desde el navegador: [apartment-rent-predictor.streamlit.app](https://apartment-rent-predictor.streamlit.app)

1. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Desde la carpeta raíz del proyecto, lanzar la aplicación:

   ```bash
   streamlit run app/app.py
   ```

   Si el comando `streamlit` no se reconoce, se puede usar `python -m streamlit run app/app.py`

## Estructura del proyecto

```
Apartment-rent-predictor/
├── data/
│   ├── raw/
│   │   ├── apartments_for_rent_classified_10K.csv   # Dataset original (entrenamiento)
│   │   └── apartments_for_rent_classified_100K.csv  # Dataset completo (no incluido, descargar desde UCI)
│   └── clean/
│       ├── apartments_clean.csv                     # Dataset limpio
│       └── cities.csv                               # Coordenadas medias y estado por ciudad
├── notebooks/
│   ├── data_exploration.ipynb                       # Exploración y limpieza
│   ├── modeling.ipynb                               # Feature engineering, modelado y ajuste
│   └── new_data_test.ipynb                          # Prueba del modelo con datos nuevos (100.000 anuncios)
├── scrips/
│   ├── cleaning.py                                  # Funciones de limpieza
│   └── modeling.py                                  # Funciones de feature engineering, evaluación y predicción
├── models/
│   ├── rent_model.pkl                               # Modelo final (Random Forest)
│   └── model_columns.pkl                            # Columnas que espera el modelo
├── app/
│   └── app.py                                       # Aplicación de Streamlit
├── presentation/
│   └── Apartment_Rent_Predictor.pptx                # Presentación del proyecto
├── requirements.txt                                 # Librerías necesarias
└── README.md
```

## Limitaciones

- Los datos son de 2019 (previos a la pandemia), por lo que el modelo refleja precios de ese momento.
- La distribución geográfica está desequilibrada: Texas y California concentran más de una cuarta parte de los anuncios, y muchas ciudades tienen muy pocos datos.
- Las viviendas de gama alta son escasas y el modelo comete errores mayores con ellas.
- Grid Search y Random Search no garantizan los parámetros óptimos, sino los mejores de entre los probados.
- La aplicación no comprueba si los datos introducidos son coherentes (por ejemplo, 5 dormitorios en 75 m²): el modelo siempre devuelve una estimación.

## Próximos pasos

- **Datos actuales:** reentrenar el modelo con anuncios recientes para reflejar el mercado posterior a la pandemia.
- **Más datos:** entrenar con la versión completa del dataset (100.000 anuncios) para mejorar la fiabilidad en ciudades y estados con pocos anuncios.
- **Nuevas variables:** incorporar información urbana, como la distancia al centro de la ciudad, al transporte público o a servicios, y la antigüedad del edificio.
- **Servicios individuales:** analizar qué servicios concretos (ascensor, piscina, parking...) influyen más en el precio, en lugar de usar solo su número total.
- **Validación en la aplicación:** avisar al usuario cuando los datos introducidos no sean coherentes.
- **Métodos espaciales:** aplicar técnicas propias del análisis urbano, como la econometría espacial o la regresión geográficamente ponderada (GWR), y evaluar el modelo con validación cruzada espacial para medir su rendimiento en mercados no vistos.

##  Autor

**Juan Carlos Del Mar Lostanau**
Arquitecto (Perú) con estudios de máster en Planeamiento Urbano, especialidad en Estudios Urbanos. Estudiante de Data Analytics en Ironhack, con interés en la aplicación de datos y Machine Learning al sector inmobiliario y las ciudades inteligentes.