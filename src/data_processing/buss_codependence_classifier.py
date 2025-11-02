# Copyright 2025 José Arturo Ornelas Brand

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# --- PARÁMETROS ---
FILE_NAME = 'buss_projections_for_classification.csv'
N_AXES = 10
RANDOM_STATE = 42

def run_buss_classifier():
    """ Ejecuta el clasificador de sentimiento usando solo los ejes BUSS. """
    print("--- FASE 4: Clasificador de Codependencia BUSS ---")
    
    # 1. Cargar datos
    try:
        data = pd.read_csv(FILE_NAME)
        print(f"Datos cargados: {data.shape} (Documentos x Columnas)")
    except FileNotFoundError:
        print(f"ERROR: Archivo {FILE_NAME} no encontrado.")
        return

    # --- CORRECCIÓN CLAVE ---
    # Usamos los nombres reales de las columnas del CSV
    feature_cols = [f'BUSS_AXIS_{i+1}' for i in range(N_AXES)]
    target_col = 'SENTIMENT_LABEL'
    # -------------------------
    
    # 2. Preparación de datos (Separar X y y)
    try:
        X = data[feature_cols] # Los 10 Ejes BUSS
        y = data[target_col]   # La etiqueta de sentimiento (0 o 1)
    except KeyError:
        # Esto ocurre si el archivo tiene un formato inesperado
        print("\nERROR: Nombres de columnas no coinciden con 'BUSS_AXIS_X' o 'SENTIMENT_LABEL'.")
        print(f"Columnas encontradas: {data.columns.tolist()}")
        return

    # Dividir en entrenamiento y prueba (80% train, 20% test)
    # Usamos stratify=y para asegurar una distribución equitativa de 0s y 1s
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Conjunto de Entrenamiento: {X_train.shape[0]} muestras.")
    print(f"Conjunto de Prueba: {X_test.shape[0]} muestras.")
    
    # 3. Entrenar el modelo (Regresión Logística como base)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    print("\nEntrenando Regresión Logística con 10 Ejes BUSS...")
    model.fit(X_train, y_train)

    # 4. Evaluar
    y_pred = model.predict(X_test)
    
    # Métrica Clave
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n=======================================================")
    print("      RESULTADOS DEL CLASIFICADOR BUSS (10 EJES)")
    print("=======================================================")
    print(f"Precisión (Accuracy): {accuracy * 100:.2f}%")
    
    if accuracy >= 0.85:
        print("🏆 ¡META LOGRADA! Precisión > 85%. Los ejes BUSS son altamente eficientes.")
    else:
        print("⚠️ Precisión por debajo de la meta. Se requiere análisis de ejes.")

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    print("=======================================================")
    
    # 5. Analizar la importancia de los Ejes (Coeficientes)
    print("\nImportancia (Coeficientes Absolutos) de los Ejes:")
    # La magnitud (valor absoluto) del coeficiente muestra la importancia predictiva
    coefs = pd.Series(model.coef_[0], index=feature_cols).abs().sort_values(ascending=False)
    print(coefs)


if __name__ == "__main__":
    run_buss_classifier()
