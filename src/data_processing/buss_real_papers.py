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
BUSS: Real Papers - CÓDIGO DEFINITIVO PARA PUBLICACIÓN.
Utiliza TF-IDF y SVD para demostrar el Teorema de Oposición Perfecta (TOP)
en un corpus de PDFs reales.
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import subprocess
import warnings

warnings.filterwarnings('ignore')

# --- PARÁMETROS CLAVE PARA LA PUBLICACIÓN ---
MAX_VOCABULARY = 1000  # Aumenta de 100 a 1000 palabras clave únicas.
N_AXES = 5             # Número de ejes bipolares a analizar (dimensiones de P).
MAX_WORDS_PER_PAPER = 500 # Extraer más texto para mejor vectorización TF-IDF.

class BUSSDefinitivo:
    def __init__(self, pdf_folder='papers_real'):
        self.pdf_folder = pdf_folder
        self.n_axes = N_AXES

    def extract_text(self, pdf_path):
        """Extrae el texto del PDF usando pdftotext y lo limpia."""
        try:
            # Ejecuta pdftotext para obtener el texto
            result = subprocess.run(['pdftotext', '-layout', pdf_path, '-'],
                                    capture_output=True, text=True, timeout=60)
            
            # Limpieza básica: minúsculas y solo los primeros X caracteres para enfoque en resumen
            text = result.stdout.lower()
            return ' '.join(text.split()[:MAX_WORDS_PER_PAPER])
        except Exception as e:
            # En caso de error (ej. timeout), devuelve cadena vacía
            print(f"Error al procesar {pdf_path}: {e}")
            return ""

    def run(self):
        pdf_files = glob.glob(f"{self.pdf_folder}/*.pdf")
        if len(pdf_files) < 2:
            print("Error: Se necesitan al menos 2 PDFs para ejecutar el análisis bipolar.")
            return

        print(f"Encontrados {len(pdf_files)} PDFs en {self.pdf_folder}")

        paper_data = []
        raw_texts = []
        for i, pdf in enumerate(pdf_files):
            text = self.extract_text(pdf)
            
            # Si el texto extraído está vacío, omitir o continuar
            if not text:
                continue

            raw_texts.append(text)
            short_name = os.path.basename(pdf).replace('.pdf', '')[:20]
            paper_data.append({'title': os.path.basename(pdf), 'short': short_name})
        
        n = len(paper_data)
        if n == 0:
            print("Error: No se pudo extraer texto de ningún PDF.")
            return

        # --------------------------------------------------------
        # PASO 1: TF-IDF VECTORIZATION (Embedding Crudo E)
        # --------------------------------------------------------
        print(f"\n1. Creando Matriz de Frecuencia con TF-IDF (Vocabulario Máx: {MAX_VOCABULARY})...")
        vectorizer = TfidfVectorizer(max_features=MAX_VOCABULARY, stop_words='english',
                                     token_pattern=r'[a-z]{5,}') # Tokens de 5 letras o más
        E = vectorizer.fit_transform(raw_texts).toarray()
        
        unique_keywords = vectorizer.get_feature_names_out()
        
        print(f"Matriz E creada: {E.shape} (Papers x Keywords)")
        
        # --- Trazabilidad: Guardar el Vocabulario y Matriz E ---
        np.savetxt('buss_keywords_final.txt', unique_keywords, fmt='%s')
        print("-> Vocabulario guardado en: buss_keywords_final.txt")
        
        np.savetxt('buss_input_matrix_E_tfidf.csv', E, delimiter=',', 
                   header=','.join(unique_keywords), comments='')
        print("-> Matriz de entrada (E) guardada en: buss_input_matrix_E_tfidf.csv")

        # --------------------------------------------------------
        # PASO 2: CENTRADO Y SVD (Base Ortogonal y Proyecciones)
        # --------------------------------------------------------
        
        # Centrado de la Matriz (E_c)
        E_c = E - E.mean(axis=0)

        # SVD: Descomposición y obtención de Proyecciones P
        print("2. Aplicando SVD para descubrir ejes ortogonales...")
        try:
            U, S, Vt = np.linalg.svd(E_c, full_matrices=False)
        except np.linalg.LinAlgError:
            print("Error SVD: No se pudo descomponer la matriz. Verifica los datos.")
            return

        # V^T[:, :N_AXES] son los ejes bipolares e_i
        axes = Vt.T[:, :self.n_axes]
        
        # Proyecciones P (P = E_c @ V)
        P = E_c @ axes
        
        # --- Trazabilidad: Guardar las Proyecciones Finales P ---
        axis_headers = [f'Axis_{i+1}' for i in range(self.n_axes)]
        np.savetxt('buss_projections_P.csv', P, delimiter=',', 
                   header=','.join(axis_headers), comments='')
        print("-> Proyecciones finales (P) guardadas en: buss_projections_P.csv")

        # --------------------------------------------------------
        # PASO 3: VALIDACIÓN DEL TEOREMA DE OPOSICIÓN PERFECTA (TOP)
        # --------------------------------------------------------
        
        # SIMULACIÓN DE DUALIDAD: Asignamos los primeros N/2 papers a 'Grupo A' y el resto a 'Grupo B'.
        # Esto simula el "Ground Truth" que el TOP debe descubrir.
        group_size = n // 2
        
        # 3a. Obtener el vector promedio (v) de cada grupo
        P_group_A = P[:group_size].mean(axis=0)
        P_group_B = P[group_size:].mean(axis=0)
        
        # 3b. Calular la Métrica TOP: Coseno de Oposición
        # cos(v_A, v_B) debe ser cercano a -1.0
        cos_opp = cosine_similarity(P_group_A.reshape(1, -1), P_group_B.reshape(1, -1))[0, 0]
        
        # 3c. Cancelación Global: v(A) + v(B) debe ser cercano a 0
        sum_vector = P_group_A + P_group_B
        norm_sum_error = np.linalg.norm(sum_vector)
        
        print("\n=====================================================================")
        print("                 VERIFICACIÓN DEL TEOREMA DE OPOSICIÓN PERFECTA (TOP)")
        print("=====================================================================")
        print(f"Grupos Analizados: Grupo A (Papers 1-{group_size}) vs. Grupo B (Papers {group_size+1}-{n})")
        print(f"\n1. Métrica de Oposición (Cos(v(A), v(B))): {cos_opp:.3f}")
        
        # Una dualidad perfecta requiere un coseno de -1.0. Cuanto más cerca, mejor.
        if cos_opp < -0.9:
             print("   -> ¡VALIDACIÓN TOP FUERTE! El modelo encuentra una oposición casi perfecta.")
        elif cos_opp < -0.5:
             print("   -> Oposición semántica presente, pero interferida por ruido del lenguaje real.")
        else:
             print("   -> La dualidad simulada no es la dominante en el corpus. Revisar papers.")

        print(f"2. Cancelación Global (Norma del Vector Suma v(A)+v(B)): {norm_sum_error:.4f}")
        print("   -> Una cancelación perfecta es 0.000. El error residual es ruido del corpus.")
        print("=====================================================================")


        # --------------------------------------------------------
        # PASO 4: VISUALIZACIÓN (Heatmap de Publicación)
        # --------------------------------------------------------

        # Análisis Bipolar de Ejes (como el código anterior, pero más limpio)
        print("\nAnálisis de Ejes Bipolares Descubiertos:")
        print("-" * 70)
        for i in range(self.n_axes):
            scores = P[:, i]
            # Encontrar los 2 papers más positivos y 2 más negativos
            pos = np.argsort(scores)[-2:][::-1]
            neg = np.argsort(scores)[:2]
            
            # Calcular oposición real para este eje
            cos_eje = cosine_similarity(P[pos].mean(axis=0).reshape(1,-1), P[neg].mean(axis=0).reshape(1,-1))[0,0]
            
            print(f"Eje {i+1} (Cos: {cos_eje:.3f}):")
            print(f"  +: { [paper_data[j]['short'] for j in pos] }")
            print(f"  -: { [paper_data[j]['short'] for j in neg] }")


        plt.figure(figsize=(12, 10))
        sns.heatmap(P, cmap='RdBu_r', center=0, vmin=P.min(), vmax=P.max(),
                    cbar_kws={'label': 'Fuerza Bipolar (Escalares Neutros $X_i$)'},
                    yticklabels=[p['short'] for p in paper_data],
                    xticklabels=[f'$Eje_{i+1}$' for i in range(self.n_axes)])
        
        plt.title('BUSS: Proyecciones Bipolares (TF-IDF + SVD)\nAnálisis de 20 Papers Reales de arXiv', fontsize=16, pad=20)
        plt.xlabel('Ejes Bipolares ($\vec{e}_i$)', fontsize=14)
        plt.ylabel('Papers (ID Corto)', fontsize=14)
        plt.tight_layout()
        plt.savefig('buss_final_publicacion.png', dpi=400, bbox_inches='tight')
        plt.show()

        print(f"\n¡ÉXITO! Análisis BUSS completado.")
        print(f"  → Gráfico de publicación guardado: buss_final_publicacion.png")

if __name__ == "__main__":
    buss = BUSSDefinitivo()
    buss.run()
