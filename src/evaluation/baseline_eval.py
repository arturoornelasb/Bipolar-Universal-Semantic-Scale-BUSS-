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
Evaluación Cuantitativa del MODELO BASE (BASELINE)

Este script carga el modelo BASE ("microsoft/DialoGPT-small")
y mide la distancia semántica entre sus generaciones estándar y bipolares,
SIN el adaptador LoRA.

El resultado de este script es el *Baseline Score* (puntuación de referencia),
que debe ser COMPARADO con el Score de BUSS-LoRA (0.4858 y 0.5492).

Si el Score del Baseline es mayor que el de BUSS-LoRA,
el entrenamiento fue un éxito.
"""

# Dependencias: pip install torch transformers peft sentence-transformers numpy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

# --- MODELO BASE: SIN CARGAR NINGÚN ADAPTADOR LO-RA ---
BASE_MODEL = "microsoft/DialoGPT-small"

# Usamos el mismo modelo SBERT para medir la similitud semántica.
EVAL_METRIC_MODEL = 'all-MiniLM-L6-v2'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Los mismos prompts de evaluación para una comparación justa.
TEST_PAIRS = [
    ("This movie was absolutely fantastic", "This movie was absolutely terrible"),
    ("The acting was incredible and moving", "The acting was wooden and unbelievable"),
    ("I loved every minute of this film", "I hated every minute of this film"),
    ("A masterpiece of cinema", "A complete waste of time and money"),
    ("The plot was brilliant and engaging", "The plot was boring and made no sense")
]

# Parámetros de generación (los mismos que en la evaluación LoRA)
generation_args = {
    "max_length": 40,
    "num_beams": 5,
    "no_repeat_ngram_size": 2,
    "early_stopping": True
}

# =============================================================================
# 2. FUNCIÓN PRINCIPAL DE EVALUACIÓN
# =============================================================================

def evaluate_baseline_model():
    
    # --- 1. Cargar Modelo de Evaluación (SBERT) ---
    print(f"--- 1. Cargando métrica de evaluación ({EVAL_METRIC_MODEL})... ---")
    try:
        eval_model = SentenceTransformer(EVAL_METRIC_MODEL, device=DEVICE)
        print("Métrica SBERT cargada.")
    except Exception as e:
        print(f"Error cargando el modelo SBERT: {e}")
        print("Por favor, ejecuta: pip install sentence-transformers")
        return

    # --- 2. Cargar SOLO el Modelo Base ---
    print(f"--- 2. Cargando modelo BASE ({BASE_MODEL})... ---")
    try:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        
        model.to(DEVICE)
        model.eval() # Poner en modo de evaluación
        print("Modelo BASE cargado y listo en modo 'eval'.")
    except Exception as e:
        print(f"Error fatal: No se pudo cargar el modelo base. Error: {e}")
        return

    # --- 3. Bucle de Generación y Evaluación ---
    print("\n--- 3. Iniciando Generación y Evaluación Cuantitativa del BASELINE... ---")
    
    generated_pairs = [] # Almacenará (texto_std, texto_opp)

    with torch.no_grad(): # Desactivar cálculo de gradientes para inferencia
        for prompt_std_base, prompt_opp_base in TEST_PAIRS:
            print(f"\n[Prompt Positivo]: {prompt_std_base}")
            
            # Asegurar que pad_token_id esté configurado para la generación
            gen_args = generation_args.copy()
            gen_args["pad_token_id"] = tokenizer.eos_token_id

            # --- Generación Estándar (Positiva) ---
            inputs_std = tokenizer(prompt_std_base, return_tensors="pt").to(DEVICE)
            outputs_std = model.generate(**inputs_std, **gen_args)
            text_std = tokenizer.decode(outputs_std[0], skip_special_tokens=True)
            print(f"  Generated (Standard): {text_std}")

            # --- Generación Bipolar Opuesta (Negativa) ---
            # ¡IMPORTANTE! El modelo base NO entiende el prefijo "Bipolar_Opposite: "
            # Por lo tanto, se lo pasamos IGUAL que al modelo LoRA para ver su respuesta "natural"
            prompt_opp_full = f"Bipolar_Opposite: {prompt_opp_base}"
            print(f"[Prompt Opuesto]: {prompt_opp_full}")
            inputs_opp = tokenizer(prompt_opp_full, return_tensors="pt").to(DEVICE)
            outputs_opp = model.generate(**inputs_opp, **gen_args)
            text_opp = tokenizer.decode(outputs_opp[0], skip_special_tokens=True)
            print(f"  Generated (Bipolar_Opposite): {text_opp}")
            
            generated_pairs.append((text_std, text_opp))

    # --- 4. Cálculo de Similitud ---
    print("\n--- 4. Calculando Similitud Semántica (SBERT)... ---")
    
    texts_std = [pair[0] for pair in generated_pairs]
    texts_opp = [pair[1] for pair in generated_pairs]

    embeddings_std = eval_model.encode(texts_std, convert_to_tensor=True)
    embeddings_opp = eval_model.encode(texts_opp, convert_to_tensor=True)

    cosine_scores = torch.diag(util.cos_sim(embeddings_std, embeddings_opp)).cpu().numpy()

    # --- 5. REPORTE FINAL ---
    print("\n=======================================================")
    print("      REPORTE DE EVALUACIÓN CUANTITATIVA (BASELINE)      ")
    print("=======================================================")
    print(f"Total de Pruebas: {len(TEST_PAIRS)}")
    
    for i in range(len(TEST_PAIRS)):
        print(f"\n[Par de Prompts {i+1}]:")
        print(f"  Similitud (Coseno SBERT): {cosine_scores[i]:.4f}")
        
    print("\n-------------------------------------------------------")
    average_score = np.mean(cosine_scores)
    print(f"SCORE PROMEDIO (Similitud del Coseno): {average_score:.4f}")

    print("\nIMPORTANTE: ESTE ES TU PUNTO DE REFERENCIA.")
    print(f"Si tu Score de BUSS-LoRA (0.4858 / 0.5492) es MENOR que este valor,")
    print("la contribución de tu modelo es demostrable.")
    print("=======================================================")


if __name__ == "__main__":
    evaluate_baseline_model()
