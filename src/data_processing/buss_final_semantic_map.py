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



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os
import glob
import re

# --- CONFIGURACIÓN ---
IMDB_BASE_DIR = "aclImdb"
MAX_SAMPLES_PER_CLASS = 2000 # Usaremos 4000 muestras en total (2000 pos + 2000 neg)
MAX_FEATURES = 5000 
N_AXES = 10
TOP_WORDS = 10 
RANDOM_STATE = 42

def load_imdb_data(base_dir, max_samples_per_class):
    """ Carga las críticas de cine y sus etiquetas (0=Negativo, 1=Positivo). """
    texts = []
    labels = []
    
    for sentiment, label in [("pos", 1), ("neg", 0)]:
        path = os.path.join(base_dir, "train", sentiment, "*.txt")
        files = glob.glob(path)
        
        # Mezclar y limitar muestras
        np.random.seed(RANDOM_STATE)
        np.random.shuffle(files) 
        
        for file_path in files[:max_samples_per_class]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    # Limpieza básica
                    text = re.sub(r'<br\s*/?>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    texts.append(text)
                    labels.append(label)
            except Exception as e:
                print(f"Advertencia: No se pudo leer {file_path}: {e}")
    
    return texts, labels

def train_and_analyze_buss(texts):
    """ Entrena el modelo BUSS (TF-IDF -> Centrado -> SVD) y analiza la Matriz V. """
    print("--- 1. Entrenando Modelo BUSS (TF-IDF + SVD) ---")
    
    # 1. TF-IDF
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english')
    E = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # 2. Centrado de Matriz (Condición BUSS)
    E_dense = E.toarray()
    E_mean = E_dense.mean(axis=0)
    E_centered = E_dense - E_mean
    
    # 3. SVD
    # Usamos n_components = 10 (los ejes que analizamos)
    svd = TruncatedSVD(n_components=N_AXES, random_state=RANDOM_STATE)
    svd.fit(E_centered)
    
    # La Matriz V (componentes) es svd.components_ (o V_T)
    V_T = svd.components_
    
    print(f"Modelo BUSS entrenado. Matriz V_T: {V_T.shape} (Ejes x Palabras)")
    return V_T, feature_names, svd.explained_variance_ratio_

def plot_explained_variance(variance_ratio):
    """ Genera el gráfico de Varianza Explicada (Fase 1). """
    cumulative_variance = np.cumsum(variance_ratio)
    
    plt.figure(figsize=(10, 6))
    
    # La proporción de varianza explicada por el SVD es la suma de los primeros 'n' componentes
    
    # Usamos un rango de 1 a N_AXES (10) para el plot
    plt.bar(range(1, N_AXES + 1), variance_ratio[:N_AXES], alpha=0.5, align='center',
            label='Varianza por Eje', color='lightblue')
    plt.plot(range(1, N_AXES + 1), cumulative_variance[:N_AXES], 'o--', color='darkred',
             label='Varianza Acumulada')
    
    # Marcar el corte en N_AXES
    variance_final = cumulative_variance[N_AXES - 1]
    plt.axvline(N_AXES, linestyle='--', color='k', label=f'{N_AXES} Ejes Seleccionados')
    plt.annotate(f'{variance_final*100:.2f}% Acumulada',
                 xy=(N_AXES, variance_final),
                 xytext=(N_AXES + 0.5, variance_final - 0.005),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.title('BUSS: Varianza Explicada por Ejes Bipolares (Validación de Dimensionalidad)')
    plt.xlabel('Número de Eje Bipolar')
    plt.ylabel('Proporción de Varianza Explicada')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--')
    plt.ylim(0, max(cumulative_variance[:N_AXES]) * 1.1)
    plt.tight_layout()
    plt.savefig('buss_final_explained_variance.png')
    plt.close()
    print("-> Gráfico de Varianza (Fase 1) guardado: buss_final_explained_variance.png")


def analyze_bipolar_axes_words(V_T, feature_names):
    """ Usa la matriz V_T para clasificar y nombrar los ejes. """
    
    print("\n" + "="*80)
    print(f"BUSS: ANÁLISIS DE LAS {MAX_FEATURES} PALABRAS CLAVE EN POLOS")
    print("="*80)
    
    for i in range(N_AXES):
        # El componente es la i-ésima fila de V_T
        component = V_T[i]
        
        # Obtener los índices de las TOP_WORDS más positivas y negativas
        top_positive_indices = component.argsort()[-TOP_WORDS:][::-1]
        top_negative_indices = component.argsort()[:TOP_WORDS]
        
        pos_words = feature_names[top_positive_indices]
        neg_words = feature_names[top_negative_indices]
        
        # Nombrar el Eje basado en el análisis anterior (Fase 2)
        if i == 2:
            axis_name = "Sentimiento Puro"
        elif i == 1:
            axis_name = "Formato/Tipo"
        else:
            axis_name = f"Eje {i+1} Genérico"

        print(f"\n--- EJE BIPOLAR {i+1}: {axis_name} ---")
        
        # 1. Opuestos Complementarios (El Eje)
        print("  OPUESTOS COMPLEMENTARIOS (El Eje):")
        print(f"    - El concepto **{pos_words[0].upper()}** se opone y complementa a **{neg_words[0].upper()}**.")
        
        # 2. Conceptos Adyacentes (Agrupación en Polos)
        print(f"  POLO POSITIVO (+): (Conceptos Adyacentes)")
        print(f"    -> {' | '.join(pos_words)}")
        
        print(f"  POLO NEGATIVO (-): (Conceptos Adyacentes)")
        print(f"    -> {' | '.join(neg_words)}")
        

def run_final_tool():
    """ Ejecución principal """
    print("Iniciando Herramienta de Mapeo Semántico Bipolar Final (BUSS)")
    
    # 1. Cargar Datos
    texts, labels = load_imdb_data(IMDB_BASE_DIR, MAX_SAMPLES_PER_CLASS)
    if not texts:
        print(f"Error: Asegúrate de que el directorio '{IMDB_BASE_DIR}' esté presente y contenga los datos de IMDB.")
        return

    # 2. Entrenar y Analizar
    V_T, feature_names, variance_ratio = train_and_analyze_buss(texts)
    
    # 3. Visualización (Fase 1)
    plot_explained_variance(variance_ratio)
    
    # 4. Análisis de Vocabulario (Fase 2)
    analyze_bipolar_axes_words(V_T, feature_names)
    
    print("\n¡ANÁLISIS SEMÁNTICO FINALIZADO!")
    print("La estructura del conocimiento ha sido mapeada.")


if __name__ == "__main__":
    run_final_tool()
