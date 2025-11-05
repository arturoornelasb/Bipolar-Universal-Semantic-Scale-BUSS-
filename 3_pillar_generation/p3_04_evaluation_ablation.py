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
# Directorios de nuestros 3 modelos entrenados
LORA_MODEL_V1_DIR = "buss_lora_model_FINAL"      # Lambda = 0.01
LORA_MODEL_V2_DIR = "buss_lora_model_lambda_0_1" # Lambda = 0.1
LORA_MODEL_V3_DIR = "buss_lora_model_lambda_1_0" # Lambda = 1.0

V4_DIR = "data_output_v4_lemma"
CLASSIFIER_FILE = "buss_v4_classifier.pkl"
N_GENERATIONS = 10 
MAX_NEW_TOKENS = 40 

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
print(f"--- Starting Pillar 3 (Final Ablation Evaluation) ---")
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
    print(f"ERROR: Files not found. {e}")
    exit()

# ===================================================================
# STEP 2: Define Evaluation Functions
# ===================================================================
def predict_sentiment(text_batch):
    """
    Usa nuestro clasificador del Pilar 2 para predecir el sentimiento (0 o 1).
    """
    try:
        tfidf_vectors = vectorizer.transform(text_batch)
        projected_features = tfidf_vectors @ V_buss_v4
        predictions = classifier.predict(projected_features)
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return np.array([0] * len(text_batch))

def generate_and_evaluate(model_name, model, tokenizer, prompts, target_sentiment):
    """
    Genera texto y lo evalúa contra un sentimiento objetivo (0 o 1).
    """
    print(f"\n--- Testing Model: {model_name} ---")
    print(f"  Target Sentiment: {'Positive (1)' if target_sentiment == 1 else 'Negative (0)'}")
    print(f"  Generating {len(prompts) * N_GENERATIONS} samples...")
    generations = []
    for prompt in tqdm(prompts, desc="  Evaluating prompts"):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=N_GENERATIONS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, 
            temperature=0.7,
            top_k=50
        )
        decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generations.extend(decoded_texts)

    print(f"  Classifying {len(generations)} generations...")
    predictions = predict_sentiment(generations)
    
    correct_predictions = np.sum(predictions == target_sentiment)
    total_predictions = len(predictions)
    adherence_rate = (correct_predictions / total_predictions) * 100
    
    print(f"  Adherence Rate: {adherence_rate:.2f}%")
    return adherence_rate

# ===================================================================
# STEP 3: Load All Models and Run Evaluation
# ===================================================================
warnings.filterwarnings("ignore", message="A decoder-only architecture is being used")

results = {}

# --- 1. Modelo Base (Sin Entrenar) ---
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, padding_side='left')
base_tokenizer.pad_token = base_tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
base_model.eval()

results["Base (Untrained)"] = {
    "Pos": generate_and_evaluate("Base (Untrained)", base_model, base_tokenizer, POSITIVE_PROMPTS, 1),
    "Neg": generate_and_evaluate("Base (Untrained)", base_model, base_tokenizer, NEGATIVE_PROMPTS, 0)
}

# --- 2. Modelo V1 (Lambda=0.01) ---
lora_model_v1 = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
lora_model_v1 = PeftModel.from_pretrained(lora_model_v1, LORA_MODEL_V1_DIR).to(device)
lora_model_v1.eval()
lora_tokenizer_v1 = AutoTokenizer.from_pretrained(LORA_MODEL_V1_DIR, padding_side='left')

results["LoRA (Lambda=0.01)"] = {
    "Pos": generate_and_evaluate("LoRA (Lambda=0.01)", lora_model_v1, lora_tokenizer_v1, POSITIVE_PROMPTS, 1),
    "Neg": generate_and_evaluate("LoRA (Lambda=0.01)", lora_model_v1, lora_tokenizer_v1, NEGATIVE_PROMPTS, 0)
}

# --- 3. Modelo V2 (Lambda=0.1) ---
lora_model_v2 = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
lora_model_v2 = PeftModel.from_pretrained(lora_model_v2, LORA_MODEL_V2_DIR).to(device)
lora_model_v2.eval()
lora_tokenizer_v2 = AutoTokenizer.from_pretrained(LORA_MODEL_V2_DIR, padding_side='left')

results["LoRA (Lambda=0.1)"] = {
    "Pos": generate_and_evaluate("LoRA (Lambda=0.1)", lora_model_v2, lora_tokenizer_v2, POSITIVE_PROMPTS, 1),
    "Neg": generate_and_evaluate("LoRA (Lambda=0.1)", lora_model_v2, lora_tokenizer_v2, NEGATIVE_PROMPTS, 0)
}

# --- 4. Modelo V3 (Lambda=1.0) ---
lora_model_v3 = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
lora_model_v3 = PeftModel.from_pretrained(lora_model_v3, LORA_MODEL_V3_DIR).to(device)
lora_model_v3.eval()
lora_tokenizer_v3 = AutoTokenizer.from_pretrained(LORA_MODEL_V3_DIR, padding_side='left')

results["LoRA (Lambda=1.0)"] = {
    "Pos": generate_and_evaluate("LoRA (Lambda=1.0)", lora_model_v3, lora_tokenizer_v3, POSITIVE_PROMPTS, 1),
    "Neg": generate_and_evaluate("LoRA (Lambda=1.0)", lora_model_v3, lora_tokenizer_v3, NEGATIVE_PROMPTS, 0)
}

# ===================================================================
# STEP 4: Final Verdict
# ===================================================================
print("\n\n--- Pillar 3 Final Verdict: Adherence to Sentiment (Ablation Study) ---")
print("                          |   Base Model   |   LoRA (λ=0.01)  |   LoRA (λ=0.1)   |   LoRA (λ=1.0)")
print("--------------------------|----------------|------------------|------------------|------------------")
print(f" Adherencia Positiva (%)  |   {results['Base (Untrained)']['Pos']:<12.2f} |   {results['LoRA (Lambda=0.01)']['Pos']:<16.2f} |   {results['LoRA (Lambda=0.1)']['Pos']:<16.2f} |   {results['LoRA (Lambda=1.0)']['Pos']:<16.2f}")
print(f" Adherencia Negativa (%)  |   {results['Base (Untrained)']['Neg']:<12.2f} |   {results['LoRA (Lambda=0.01)']['Neg']:<16.2f} |   {results['LoRA (Lambda=0.1)']['Neg']:<16.2f} |   {results['LoRA (Lambda=1.0)']['Neg']:<16.2f}")
print("-------------------------------------------------------------------------------------------------")
print("Tasa de Adherencia = % de generaciones que el clasificador del Pilar 2")
print("                   consideró que coincidían con el sentimiento del prompt.")
print("\n--- Pillar 3 Complete (Definitive) ---")