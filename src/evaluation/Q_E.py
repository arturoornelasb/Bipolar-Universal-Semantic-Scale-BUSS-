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



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# =============================================================================
# 1. FUNCIÓN DE EMBEDDING PARA EVALUACIÓN (SIMULANDO BUSS)
# =============================================================================

def get_buss_embeddings(texts, original_texts=None, n_components=24):
    """
    Función de utilidad para generar embeddings y comparar. 
    Utiliza un vectorizador TF-IDF y Truncated SVD, similar a BUSS, 
    pero simplificado para la evaluación.
    """
    if original_texts is None:
        # Usar los textos proporcionados como corpus base si no se da un corpus original
        original_texts = texts
        
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    # 1. Ajustar (Fit) en el corpus original y transformar
    E_orig = vectorizer.fit_transform(original_texts).toarray()
    
    # 2. Centrar (usando la media del corpus original)
    E_c_orig = E_orig - E_orig.mean(axis=0)

    # 3. Ajustar SVD en los datos centrados del corpus original
    svd = TruncatedSVD(n_components=min(n_components, E_c_orig.shape[1] - 1))
    svd.fit(E_c_orig)
    
    # 4. Transformar los textos de entrada (generados)
    E_texts = vectorizer.transform(texts).toarray()
    E_texts_c = E_texts - E_orig.mean(axis=0) # Centrar con la media original
    P_texts = svd.transform(E_texts_c)
    
    # 5. Normalizar para la Similitud del Coseno
    norms = np.linalg.norm(P_texts, axis=1, keepdims=True)
    # Evitar división por cero
    P_norm = np.where(norms > 1e-6, P_texts / norms, P_texts)
    
    return P_norm

# =============================================================================
# 2. CONFIGURACIÓN Y PRUEBAS
# =============================================================================

def run_quantitative_eval():
    # Rutas del modelo y tokenizer guardados
    model_path = "./buss_lora_final"
    base_model_name = "microsoft/DialoGPT-small"

    # --- 1. Cargar el Tokenizer y el Modelo Base ---
    print("--- 1. Cargando Tokenizer y Modelo Base... ---")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # --- 2. Cargar el Adaptador LoRA ---
    print("--- 2. Cargando Adaptador LoRA... ---")
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
    except Exception as e:
        print(f"Error al cargar el modelo LoRA desde {model_path}. Asegúrese de que la carpeta existe.")
        print(f"Detalle: {e}")
        return

    # Mover el modelo a la GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Modelo cargado y movido a: {device}")

    # --- 3. Definir Conjunto de Pruebas ---
    test_prompts = [
        "The impact of Artificial Intelligence is",
        "Climate change mitigation requires",
        "A healthy balance sheet shows",
        "In the realm of modern art,",
        "The political decision led to"
    ]
    
    # Usar el corpus de entrenamiento original para el vectorizador
    original_texts_for_buss_fit = [
        "Machine learning uses embeddings for semantic similarity.",
        "Neural networks learn hierarchical representations.",
        "SVD decomposes matrices into orthogonal components.",
        "Bipolar scales capture opposition in language.",
        "LoRA enables efficient fine-tuning of LLMs."
    ] * 5

    generated_texts_standard = []
    generated_texts_opposite = []
    
    # --- 4. Generación Estándar y Opuesta ---
    print("\n--- 4. Generando textos Estándar y Opuestos... ---")
    
    for prompt in test_prompts:
        # Generación Estándar
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs, 
            max_length=50, 
            num_return_sequences=1,
            do_sample=True, # Usar muestreo para mayor diversidad
            top_k=50,
            temperature=0.7,
        )
        std_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        generated_texts_standard.append(std_text)
        
        # Generación Opuesta (Usando el prefijo aprendido)
        opp_prompt = f"Bipolar_Opposite: {prompt}"
        inputs_opp = tokenizer(opp_prompt, return_tensors="pt").to(device)
        outputs_opp = model.generate(
            **inputs_opp, 
            max_length=50, 
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            temperature=0.7,
        )
        opp_text = tokenizer.decode(outputs_opp[0], skip_special_tokens=True).strip()
        # Aseguramos que el prefijo se elimine del texto para la comparación de embeddings
        opp_text = opp_text.replace("Bipolar_Opposite:", "").strip() 
        generated_texts_opposite.append(opp_text)
        
        print(f"\n[Prompt]: {prompt}")
        print(f"  [Estándar]: {std_text}")
        print(f"  [Opuesto]: {opp_text}")


    # --- 5. Cálculo de Similitud del Coseno (Métrica de Contraste) ---
    print("\n--- 5. Calculando Similitud del Coseno... ---")
    
    all_generated_texts = generated_texts_standard + generated_texts_opposite
    
    # Generar embeddings para todos los textos generados, usando el corpus original
    # para la contextualización del BUSS-like embedding.
    all_embeddings = get_buss_embeddings(all_generated_texts, original_texts_for_buss_fit)
    
    # Dividir de nuevo en Standard y Opposite embeddings
    num_tests = len(test_prompts)
    std_embeddings = all_embeddings[:num_tests]
    opp_embeddings = all_embeddings[num_tests:]
    
    # Calcular la Similitud del Coseno par a par
    cosine_scores = [
        cosine_similarity(std_embeddings[i].reshape(1, -1), opp_embeddings[i].reshape(1, -1))[0][0]
        for i in range(num_tests)
    ]
    
    # --- 6. Reporte Final ---
    print("\n=======================================================")
    print("        REPORTE DE EVALUACIÓN CUANTITATIVA (BUSS)       ")
    print("=======================================================")
    print(f"Total de Pruebas: {num_tests}")
    
    results = []
    for i in range(num_tests):
        results.append({
            "Prompt": test_prompts[i],
            "Score": cosine_scores[i],
            "Std_Text": generated_texts_standard[i],
            "Opp_Text": generated_texts_opposite[i]
        })
        print(f"\n[Prompt]: {test_prompts[i]}")
        print(f"  Similitud (Coseno): {cosine_scores[i]:.4f}")
        print(f"  Resultado Esperado: Score CERCANO a 0 o NEGATIVO.")

    avg_score = np.mean(cosine_scores)
    print("\n-------------------------------------------------------")
    print(f"SCORE PROMEDIO (Similitud del Coseno Estándar vs. Opuesto): {avg_score:.4f}")
    
    if avg_score < 0.1:
        print("🎉 ¡ÉXITO! El puntaje promedio es bajo (cercano a cero o negativo).")
        print("Esto indica que el modelo BUSS-LoRA ha aprendido a generar conceptos")
        print("semánticamente distantes (opuestos) cuando se le indica con el prefijo.")
    else:
        print("⚠️ ALERTA: El puntaje promedio es alto. Esto sugiere que las generaciones")
        print("Estándar y Opuesta aún son semánticamente similares.")
        print("Se recomienda aumentar el 'bipolar_weight' o los 'num_train_epochs'.")
    print("=======================================================")

if __name__ == "__main__":
    run_quantitative_eval()
