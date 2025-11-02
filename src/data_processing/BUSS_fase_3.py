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

#!/usr/bin/env python3
"""
Análisis de Ejes Bipolares BUSS (Fase 3)

Este script implementa la visión original de BUSS: analizar un corpus
grande (IMDB) para descubrir los ejes semánticos (dualidades)
subyacentes que lo definen.
"""

# Dependencias: pip install scikit-learn numpy pandas
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
import os
import glob
import re

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CARGADOR DE DATOS (IMDB REAL - SIN PREFIJOS)
# =============================================================================

def load_imdb_data_raw(base_dir, max_samples_per_class=2000):
    """
    Carga los archivos .txt de las carpetas train/pos y train/neg
    y devuelve una sola lista de todos los textos y las etiquetas binarias.
    """
    all_texts = []
    all_labels = []
    
    pos_dir = os.path.join(base_dir, "train", "pos")
    neg_dir = os.path.join(base_dir, "train", "neg")

    print(f"--- 1. Cargando datos crudos desde {base_dir} ---")
    
    # Cargar Textos Positivos (Label 1)
    pos_files = glob.glob(os.path.join(pos_dir, "*.txt"))
    np.random.shuffle(pos_files) 
    
    for file_path in pos_files[:max_samples_per_class]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                text = re.sub(r'<br\s*/?>', ' ', text) # Limpiar HTML
                text = re.sub(r'\s+', ' ', text).strip().lower()
                all_texts.append(text)
                all_labels.append(1) # Positivo
        except Exception:
            pass # Ignorar archivos que fallan

    # Cargar Textos Negativos (Label 0)
    neg_files = glob.glob(os.path.join(neg_dir, "*.txt"))
    np.random.shuffle(neg_files)

    for file_path in neg_files[:max_samples_per_class]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                text = re.sub(r'<br\s*/?>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip().lower()
                all_texts.append(text)
                all_labels.append(0) # Negativo
        except Exception:
            pass # Ignorar archivos que fallan

    if not all_texts:
        print(f"Error: No se encontraron archivos de texto en {pos_dir} o {neg_dir}.")
        print("¿Descargaste y descomprimiste 'aclImdb_v1.tar.gz' en esta carpeta?")
        return None, None

    print(f"Se cargaron {len(all_texts)} críticas (Positivas y Negativas).")
    return all_texts, all_labels

# =============================================================================
# 2. FUNCIÓN DE ANÁLISIS DE EJES
# =============================================================================

def analyze_buss_axes():
    
    # --- Parámetros de Análisis ---
    MAX_SAMPLES = 2000 # 2000 pos + 2000 neg = 4000 total
    MAX_FEATURES = 5000 
    N_COMPONENTS = 10   
    OUTPUT_FILE = "buss_projections_for_classification.csv"

    # --- 1. Cargar Datos ---
    all_texts, all_labels = load_imdb_data_raw(
        base_dir="aclImdb", 
        max_samples_per_class=MAX_SAMPLES
    )
    if all_texts is None:
        return

    # --- 2. Entrenar BUSS (TF-IDF + SVD Centrado) ---
    print("\n--- 2. Entrenando modelo BUSS (TF-IDF + SVD)... ---")
    
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english', max_df=0.8, min_df=5)
    E_sparse = vectorizer.fit_transform(all_texts)
    
    # Convertimos la matriz dispersa a densa para evitar np.matrix
    E = E_sparse.toarray()
    
    print(f"Matriz E creada: {E.shape[0]} Documentos, {E.shape[1]} Features (Palabras)")

    # Centrar los datos (el corazón de BUSS)
    E_mean = E.mean(axis=0)
    E_c = E - E_mean

    # Aplicar SVD
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    P = svd.fit_transform(E_c) # P es la matriz de Proyecciones

    print("Modelo BUSS entrenado. Matriz P (Proyecciones) calculada.")

    # --- 3. Guardar Proyecciones para la Clasificación (Fase 4) ---
    print(f"--- Guardando Proyecciones BUSS en {OUTPUT_FILE} ---")
    
    # Crear nombres de columna para los ejes
    column_names = [f'BUSS_AXIS_{i+1}' for i in range(N_COMPONENTS)]
    
    # Crear el DataFrame
    df = pd.DataFrame(P, columns=column_names)
    df['SENTIMENT_LABEL'] = all_labels # Añadimos la etiqueta de sentimiento (0=Neg, 1=Pos)
    
    # Guardar en CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print("ÉXITO: Matriz de Proyecciones guardada para Fase 4.")

    # --- 4. Analizar Componentes (Ejes) ---
    print("\n--- 4. ANÁLISIS DE EJES BIPOLARES (Nombrando la Codependencia) ---")
    
    terms = vectorizer.get_feature_names_out()
    axes = svd.components_

    for i in range(N_COMPONENTS):
        print(f"\n=======================================================")
        print(f"   EJE BIPOLAR {i+1} (Varianza Explicada: {svd.explained_variance_ratio_[i]:.2%})")
        print("=======================================================")
        
        axis_component = axes[i]
        
        # Obtener los índices de las palabras con más peso
        top_positive_indices = axis_component.argsort()[-10:][::-1] 
        top_negative_indices = axis_component.argsort()[:10]       
        
        # Mapear índices a palabras (terms)
        positive_keywords = [terms[idx] for idx in top_positive_indices]
        negative_keywords = [terms[idx] for idx in top_negative_indices]

        print(f"  POLO POSITIVO (+): {', '.join(positive_keywords)}")
        print(f"  POLO NEGATIVO (-): {', '.join(negative_keywords)}")

    print("\nAnálisis de ejes completado.")

if __name__ == "__main__":
    analyze_buss_axes()
