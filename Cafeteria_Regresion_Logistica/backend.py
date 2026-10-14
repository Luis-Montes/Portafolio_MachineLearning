import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from connection.connection import engine, query
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    df_clientes = pd.read_sql(query, engine)
    print("Data loaded successfully.")
    print(df_clientes.head())

    df_encode = pd.get_dummies(df_clientes, columns=["tipo_bebida", "es_estudiante"], drop_first=True)

    x = df_encode.drop(columns=["id_cliente", "gasto_total"])
    y = df_encode["es_cliente_fiel"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    model = LogisticRegression()
    model.fit(x_train, y_train)
    print("Model trained successfully.")

    y_pred = model.predict(x_test)

    print("\n--- Resultados del Modelo de Clasificación ---")
    print("Accuracy (Exactitud):", accuracy_score(y_test, y_pred))
    print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

load_data()