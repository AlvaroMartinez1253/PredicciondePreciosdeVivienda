# 1. Importaciones necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Importar Imputer
import joblib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 2. Carga de Datos (Asegúrate de tener kc_house_data.csv)
try:
    # Intenta cargar datos directamente de un repositorio público si no existe localmente
    data_url = 'https://raw.githubusercontent.com/datasets/house-prices-in-seattle/main/kc_house_data.csv'
    df = pd.read_csv(data_url)
except Exception:
    try:
        df = pd.read_csv('kc_house_data.csv')
    except FileNotFoundError:
        print("Error: Asegúrate de que 'kc_house_data.csv' esté en el directorio o accesible.")
        exit()

# --- 3. Ingeniería de Características y Limpieza ---
# Rellena valores nulos que se puedan haber creado durante la carga (si aplica)
df.fillna(0, inplace=True) # Manejo simple de nulos, se debe mejorar en un proyecto real

# Crear Features de Ingeniería
CURRENT_YEAR = 2025 # Año de referencia
df['house_age'] = CURRENT_YEAR - df['yr_built'] 
df['total_sqft'] = df['sqft_living'] + df['sqft_basement']

# 4. Preparación de X e y
# Usamos log(price) para mitigar el sesgo
df['price_log'] = np.log1p(df['price'])
# Excluir price original, ID y date. Mantenemos TODAS las demás columnas en X.
X = df.drop(['id', 'date', 'price', 'price_log'], axis=1) 
y = df['price_log']

# 5. División de Datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Definición de Columnas para Preprocesamiento
# Estas listas DEBEN coincidir con los nombres de las columnas en X_train
numerical_features = [
    'sqft_living', 'bathrooms', 'bedrooms', 'sqft_lot', 'floors', 'view', 'grade',
    'sqft_above', 'sqft_basement', 'yr_renovated', 'lat', 'long', 
    'sqft_living15', 'sqft_lot15', 'house_age', 'total_sqft', 'yr_built', 'zipcode'
]
categorical_features = ['waterfront', 'condition'] 

# 7. Creación de Preprocesador con Pipeline
# Se añade un Imputer para robustecer contra nulos si aparecen
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    # Usamos 'drop' para las columnas que no están en las listas anteriores (ninguna en este caso)
    remainder='drop' 
)

# 8. Creación del Pipeline Final (Preprocesador + Modelo)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=5))
])

# 9. Entrenamiento del Modelo
print("Entrenando el modelo...")
model.fit(X_train, y_train)
print("Entrenamiento completado.")

# 10. Evaluación del Modelo (Métricas)
y_pred_log = model.predict(X_test)
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
r2 = r2_score(y_test_real, y_pred_real)

print(f"\n--- Resultados de Evaluación ---")
print(f"RMSE (Error Típico de Predicción): ${rmse:,.2f}")
print(f"R^2 (Varianza Explicada): {r2:.4f}")

# 11. Guardar el Modelo y las Columnas
# Guardamos el modelo (Pipeline) y la lista de nombres de columnas usadas
artifacts = {
    'model': model,
    'features': list(X_train.columns) # **GUARDAMOS LA LISTA DE COLUMNAS**
}

joblib.dump(artifacts, 'modelo_regresion_vivienda_artifacts.pkl')
print("\nModelo y lista de features guardados correctamente como 'modelo_regresion_vivienda_artifacts.pkl'")