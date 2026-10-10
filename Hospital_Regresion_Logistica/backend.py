# El modelo de clasificación
from sklearn.linear_model import LinearRegression
import pandas as pd

# Las métricas de evaluación para clasificación
from sklearn.metrics import (
    mean_squared_error,  # Error Cuadrático Medio (magnitud del error)
    r2_score,  # Coeficiente de Determinación (qué tan bien se ajusta el modelo)
)
from sklearn.model_selection import train_test_split

def load_data():
    df_insurance = pd.read_csv("insurance.csv")
    print("Datos cargados correctamente:")
    print(df_insurance.head())

    df_encode = pd.get_dummies(
        df_insurance, columns=["sex", "smoker", "region"], drop_first=True
    )

    X = df_encode.drop(columns=["charges"])
    Y = df_encode["charges"]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print("Datos divididos en entrenamiento y prueba:")

    model = LinearRegression()
    model.fit(x_train, y_train)
    print("Modelo entrenado correctamente.")

    y_pred = model.predict(x_test)

    print("\n--- Resultados del Modelo de Regresión Lineal ---")
    print("Error Cuadrático Medio (MSE):", mean_squared_error(y_test, y_pred))
    print("R² Score (Precisión del modelo):", r2_score(y_test, y_pred))

load_data()