🏡 Predicción de Precios de Viviendas en King County, WA
Este proyecto desarrolla un modelo de Machine Learning de Regresión para predecir el precio de venta de casas en King County, Washington (área de Seattle), utilizando datos históricos. El objetivo es construir el modelo con el mejor desempeño predictivo y documentar un pipeline robusto, listo para producción.

🎯 Objetivo y Metodología
Problema de Negocio
Estimar el valor de una vivienda en el mercado, proporcionando una herramienta de predicción para tasadores o posibles compradores/vendedores.

Solución ML
Se utiliza la Regresión para predecir una variable continua (price). El proceso se basa en un Pipeline de Scikit-learn para asegurar que el preprocesamiento de datos se aplique de manera consistente.

📊 Análisis Exploratorio de Datos (EDA)
El EDA fue crucial para entender la distribución de la variable objetivo y las relaciones con las variables predictoras.

Transformación de la Variable Objetivo (price): El precio mostró una distribución altamente sesgada a la derecha. Para estabilizar la varianza y mejorar el ajuste del modelo, se aplicó una transformación logarítmica (log(1+price)).

Correlaciones Clave: Se identificaron las variables más correlacionadas con el precio:

sqft_living (Área habitable)

grade (Grado de la casa, calidad de construcción)

bathrooms y house_age (Edad de la casa).

Manejo de Outliers: Se investigaron y mitigaron outliers extremos en el área habitable y el precio que podían distorsionar el modelo.

⚙️ Preprocesamiento y Diseño del Pipeline
Se construyó un ColumnTransformer dentro de un Pipeline para manejar distintos tipos de features simultáneamente:

Variables Numéricas: Aplicación de StandardScaler para normalizar los datos (media 0, desviación estándar 1).

Variables Categóricas: Aplicación de OneHotEncoder a features como waterfront y condition.

Ingeniería de Características: Creación de la feature house_age (2025 - yr_built) para capturar el valor de la antigüedad de la propiedad, lo cual es más relevante que el año de construcción en sí.

🧠 Modelado y Evaluación
Se compararon dos modelos principales.

Modelo	Métrica	RMSE	R 
2
 	Observaciones
Regresión Lineal (Baseline)	y vs  
y
^
​
 	$225,480	0.68	Modelo simple, sensible a relaciones no lineales.
Random Forest Regressor	y vs  
y
^
​
 	$118,912	0.89	Modelo Ganador. Maneja bien la complejidad y las interacciones de features.

Exportar a Hojas de cálculo
El modelo Random Forest Regressor fue elegido como el modelo final debido a su capacidad para reducir significativamente el RMSE (Error Raíz Cuadrático Medio), lo que indica un error de predicción promedio sustancialmente menor.

RMSE de $118,912 significa que, en promedio, la predicción del modelo se desvía del precio real en esa cantidad de dólares.

🚀 Despliegue (Deployment)
El modelo entrenado y el pipeline completo se serializaron usando joblib y se desplegaron como una aplicación web interactiva.

Estructura de Archivos
├── README.md
├── kc_house_data.csv
├── modelo.py              # Script de entrenamiento y guardado del modelo
├── app.py                 # Aplicación Streamlit para la interfaz
├── modelo_regresion_vivienda.pkl # Objeto Pipeline guardado (modelo)
└── requirements.txt       # Librerías y versiones para el deployment
Cómo Ejecutar la Aplicación
Clonar el repositorio:

Bash

git clone [TU_URL_DE_GITHUB]
cd [nombre_del_repo]
Crear y activar un entorno virtual.

Instalar dependencias:

Bash

pip install -r requirements.txt
Ejecutar la aplicación Streamlit:

Bash

streamlit run app.py