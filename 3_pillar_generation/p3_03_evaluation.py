# Copyright 2025 José Arturo Ornelas Brand

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pickle
import numpy as np
import os
from scipy.sparse import load_npz
from tqdm import tqdm
import warnings

# --- Configuration ---
BASE_MODEL_NAME = "microsoft/DialoGPT-small"
LORA_MODEL_DIR = "buss_lora_model_FINAL"
V4_DIR = "data_output_v4_lemma"
CLASSIFIER_FILE = "buss_v4_classifier.pkl"
N_GENERATIONS = 10 # Número de textos a generar por prompt
MAX_NEW_TOKENS = 40 # Longitud de cada generación

# --- Define Test Prompts ---
POSITIVE_PROMPTS = [
    "I loved this movie, it was",
    "This is one of the best films I've ever seen, I",
    "Absolutely wonderful! The acting was",
    "A masterpiece of cinema. The story was",
    "I highly recommend this, it's a"
]
NEGATIVE_PROMPTS = [
    "I hated this movie, it was",
    "This is one of the worst films I've ever seen, I",
    "Absolutely terrible! The acting was",
    "A disaster of cinema. The story was",
    "I do not recommend this, it's a"
]

# --- CUDA Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Starting Pillar 3 (Final Evaluation) ---")
print(f"Using device: {device}")

# ===================================================================
# STEP 1: Load Artifacts (Classifier, Vectorizer, Axes)
# ===================================================================
print("Step 1: Loading Classifier and BUSS Artifacts...")

# --- Define the tokenizer function (required for pickle) ---
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
import nltk.corpus 
stop_words = set(nltk.corpus.stopwords.words('english'))
def lemma_tokenizer(text):
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            lemmatized_tokens.append(lemmatizer.lemmatize(token))
    return lemmatized_tokens
# --- End of pickle fix ---

try:
    with open(os.path.join(V4_DIR, "tfidf_vectorizer_v4.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    
    with open(CLASSIFIER_FILE, "rb") as f:
        classifier = pickle.load(f)
        
    V_buss_v4 = np.load(os.path.join(V4_DIR, "V_buss_lemma.npy")) # (5000, 500)
    print("Classifier and BUSS artifacts loaded.")

except FileNotFoundError as e:
    print(f"ERROR: Files not found. Did you run p3_02_build_classifier.py? {e}")
    exit()

# ===================================================================
# STEP 2: Define Evaluation Functions
# ===================================================================
def predict_sentiment(text_batch):
    """
    Usa nuestro clasificador del Pilar 2 para predecir el sentimiento
    de un lote de textos generados.
    """
    try:
        # 1. Vectorizar texto
        tfidf_vectors = vectorizer.transform(text_batch)
        # 2. Proyectar sobre ejes V4
        projected_features = tfidf_vectors @ V_buss_v4
        # 3. Predecir (0=Neg, 1=Pos)
        predictions = classifier.predict(projected_features)
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return np.array([0] * len(text_batch)) # Devuelve 0 por defecto en caso de error

def generate_and_evaluate(model, tokenizer, prompts, target_sentiment):
    """
    Genera texto desde un modelo y lo evalúa contra un sentimiento objetivo.
    target_sentiment: 0 (Negativo) o 1 (Positivo)
    """
    print(f"  Generating {len(prompts) * N_GENERATIONS} samples...")
    generations = []
    for prompt in tqdm(prompts, desc="  Evaluating prompts"):
        # Tokenizar el prompt (con padding izquierdo)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        
        # Generar N veces
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=N_GENERATIONS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, # Activar muestreo para variabilidad
            temperature=0.7,
            top_k=50
        )
        
        # Decodificar y añadir a la lista
        decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generations.extend(decoded_texts)

    # Evaluar todas las generaciones de una vez
    print(f"  Classifying {len(generations)} generations...")
    predictions = predict_sentiment(generations)
    
    # Calcular Tasa de Adherencia
    # np.sum(predictions == target_sentiment)
    correct_predictions = np.sum(predictions == target_sentiment)
    total_predictions = len(predictions)
    adherence_rate = (correct_predictions / total_predictions) * 100
    
    return adherence_rate

# ===================================================================
# STEP 3: Load Models
# ===================================================================
warnings.filterwarnings("ignore", message="A decoder-only architecture is being used")

# --- Modelo 1: Base (Sin Entrenar) ---
print("\nStep 3a: Loading BASE Model (DialoGPT-small)...")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, padding_side='left')
base_tokenizer.pad_token = base_tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
base_model.eval() # Poner en modo de evaluación

# --- Modelo 2: BUSS-LoRA (Entrenado) ---
print("\nStep 3b: Loading TRAINED BUSS-LoRA Model...")
# Cargar el modelo base primero
lora_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
# Cargar el tokenizador que guardamos
lora_tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_DIR, padding_side='left')

# Cargar los adaptadores LoRA (PEFT) encima
lora_model = PeftModel.from_pretrained(lora_model, LORA_MODEL_DIR).to(device)
lora_model.eval() # Poner en modo de evaluación
print("Models loaded.")

# ===================================================================
# STEP 4: Run Final Evaluation
# ===================================================================
print("\n--- Running Evaluation: BASE MODEL (Sin Entrenar) ---")
print("\nTesting POSITIVE prompts (Target=1)...")
base_pos_rate = generate_and_evaluate(base_model, base_tokenizer, POSITIVE_PROMPTS, target_sentiment=1)

print("\nTesting NEGATIVE prompts (Target=0)...")
base_neg_rate = generate_and_evaluate(base_model, base_tokenizer, NEGATIVE_PROMPTS, target_sentiment=0)


print("\n--- Running Evaluation: BUSS-LORA MODEL (Entrenado) ---")
print("\nTesting POSITIVE prompts (Target=1)...")
lora_pos_rate = generate_and_evaluate(lora_model, lora_tokenizer, POSITIVE_PROMPTS, target_sentiment=1)

print("\nTesting NEGATIVE prompts (Target=0)...")
lora_neg_rate = generate_and_evaluate(lora_model, lora_tokenizer, NEGATIVE_PROMPTS, target_sentiment=0)


# ===================================================================
# STEP 5: Final Verdict
# ===================================================================
print("\n\n--- Pillar 3 Final Verdict: Adherencia al Sentimiento ---")
print("                          |   Base Model (Sin Entrenar)  |   BUSS-LoRA (Entrenado)")
print("--------------------------|------------------------------|-------------------------")
print(f" Adherencia Positiva      |   {base_pos_rate:<26.2f}% |   {lora_pos_rate:<23.2f}%")
print(f" Adherencia Negativa      |   {base_neg_rate:<26.2f}% |   {lora_neg_rate:<23.2f}%")
print("-----------------------------------------------------------------")
print("Tasa de Adherencia = % de generaciones que el clasificador del Pilar 2")
print("                   consideró que coincidían con el sentimiento del prompt.")
print("\n--- Pillar 3 Complete (Definitive) ---")