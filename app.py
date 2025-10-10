# 1. Importaciones para Streamlit y ML
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Definir una constante para el año actual para las features de ingeniería
CURRENT_YEAR = 2025

# 2. Carga del Modelo y las Features
ARTIFACTS_FILE = 'modelo_regresion_vivienda_artifacts.pkl'
try:
    artifacts = joblib.load(ARTIFACTS_FILE)
    model = artifacts['model']
    # Cargamos la lista de features usadas en el entrenamiento
    TRAINING_FEATURES = artifacts['features'] 
except FileNotFoundError:
    st.error(f"Error: Archivo '{ARTIFACTS_FILE}' no encontrado. Por favor, corre 'modelo.py' primero.")
    st.stop()

# 3. Configuración de la Aplicación
st.set_page_config(page_title="Predicción de Precios de Vivienda", layout="centered")

st.title("🏡 Predictor de Precios de Viviendas (Portafolio ML)")
st.markdown("Ajusta los parámetros de la vivienda para obtener una estimación de su precio.")

# 4. Definición de Parámetros de Entrada (Input Widgets)
# Solo recolectamos las features más importantes para la UI
with st.sidebar:
    st.header("Características Clave")

    sqft_living = st.slider("Área de Vivienda (sqft)", min_value=500, max_value=8000, value=2000, step=100)
    bedrooms = st.slider("Número de Habitaciones", min_value=1, max_value=8, value=3)
    bathrooms = st.slider("Número de Baños", min_value=1.0, max_value=5.0, value=2.5, step=0.5)
    floors = st.slider("Número de Pisos", min_value=1.0, max_value=3.5, value=2.0, step=0.5)
    grade = st.selectbox("Calidad de Construcción (Grado)", options=list(range(1, 14)), index=7, help="Escala de 1 (peor) a 13 (mejor). 7 es promedio.")
    yr_built = st.slider("Año de Construcción", min_value=1900, max_value=2015, value=1980)
    zipcode = st.text_input("Código Postal (Zipcode)", value="98103", help="El código postal es un predictor clave.")
    
    st.header("Otras Características")
    waterfront = st.selectbox("Frente al Agua", options=[0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No')
    condition = st.selectbox("Condición", options=[1, 2, 3, 4, 5], index=2)
    view = st.selectbox("Vista (Escala 0-4)", options=[0, 1, 2, 3, 4], index=0)
    sqft_basement = st.slider("Sótano (sqft)", min_value=0, max_value=3000, value=0, step=100)


# 5. Mapeo de Entradas a Features (La clave para la corrección)
if st.button("Calcular Precio Estimado"):
    
    # --- 5.1. Features de Ingeniería (Deben coincidir con modelo.py) ---
    house_age = CURRENT_YEAR - yr_built
    total_sqft = sqft_living + sqft_basement
    
    # --- 5.2. Crear un diccionario con todas las features esperadas ---
    # Usamos valores promedio o por defecto para las que no están en la UI 
    # (lat, long, sqft_lot, etc.)
    
    data = {
        # Features recolectadas en la UI
        'sqft_living': sqft_living,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'floors': floors,
        'grade': grade,
        'waterfront': waterfront,
        'condition': condition,
        'view': view,
        'yr_built': yr_built,
        'sqft_basement': sqft_basement,
        'zipcode': int(zipcode), # Asegurarse que sea entero
        
        # Features de Ingeniería
        'house_age': house_age,
        'total_sqft': total_sqft,
        
        # Features por defecto (usa valores promedio del dataset si los conoces)
        'sqft_lot': 15000, 
        'sqft_above': sqft_living - sqft_basement,
        'yr_renovated': 0, 
        'lat': 47.60, # Latitud promedio de Seattle
        'long': -122.20, # Longitud promedio de Seattle
        'sqft_living15': sqft_living, # Asumir el mismo tamaño
        'sqft_lot15': 15000 # Asumir el mismo tamaño de lote
    }
    
    # 5.3. Construir el DataFrame y REORDENAR las columnas
    input_df = pd.DataFrame([data])
    
    # **ESTA LÍNEA ES CRUCIAL:** Reordena el DataFrame de entrada para que las columnas
    # coincidan exactamente con la lista guardada del entrenamiento.
    input_df = input_df[TRAINING_FEATURES] 

    # 6. Predicción
    try:
        pred_log = model.predict(input_df)[0]
    except ValueError as e:
        st.error(f"Error de predicción: Verifica que todas las características estén bien mapeadas. Error: {e}")
        st.stop()
    
    # 7. Invertir la Transformación Logarítmica
    price_real = np.expm1(pred_log) 

    # 8. Mostrar Resultado
    st.success(f"### El Precio Estimado de la Vivienda es:")
    st.metric(label="Precio Estimado", value=f"${price_real:,.2f} USD")
    st.caption("Esta predicción usa un Random Forest Regressor entrenado con más de 20,000 registros. El error típico (RMSE) de este modelo debe considerarse para evaluar la precisión.")