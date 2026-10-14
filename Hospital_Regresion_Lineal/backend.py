import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def modelo_clasificacion():
    df = pd.read_csv('insurance.csv')
    print(df.head())

    df["gasto_alto"] = (df["charges"] > 15000).astype(int)

    df_encode = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

    X = df_encode.drop(columns=["charges", "gasto_alto"])
    y = df_encode["gasto_alto"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_logistico = LogisticRegression(max_iter=1000)
    modelo_logistico.fit(x_train, y_train)
    print("Modelo entrenado con éxito.")

    y_pred = modelo_logistico.predict(x_test)

    print("\n--- Resultados de la Clasificación ---")
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión global (Accuracy): {precision * 100:.2f}%")

    print("\nMatriz de Confusión (Aciertos vs Errores):")
    print(confusion_matrix(y_test, y_pred))

    # Prediccion para un nuevo paciente
    nuevo_paciente = {
        "age": [30],
        "sex": ["male"],
        "bmi": [25.0],
        "children": [1],
        "smoker": ["no"],
        "region": ["southeast"],
        "alto_riesgo": [32.0]
    }

    df_nuevo_paciente = pd.DataFrame(nuevo_paciente)

    df_nuevo_c_encode = pd.get_dummies(
        df_nuevo_paciente, columns=["sex", "smoker", "region"], drop_first=True
    )

    df_nuevo_c_encode = df_nuevo_c_encode.reindex(columns=X.columns, fill_value=0)

    prediccion_clase = modelo_logistico.predict(df_nuevo_c_encode)

    probabilidades = modelo_logistico.predict_proba(df_nuevo_c_encode)

    print("\n--- Predicción de Clasificación para el Cliente Nuevo ---")
    if prediccion_clase[0] == 1:
        print("Predicción: El cliente TIENE ALTA PROBABILIDAD de ser de GASTO ALTO.")
    else:
        print(
            "Predicción: El cliente tiene probabilidad de ser de GASTO NORMAL /"
            " BAJO."
    )

    prob_gasto_alto = probabilidades[0][1] * 100
    print(f"Probabilidad estimada de gasto alto: {prob_gasto_alto:.2f}%")

modelo_clasificacion()