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
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import os
import time
from scipy.sparse import csr_matrix
import warnings

# --- Configuration ---
MODEL_NAME = "microsoft/DialoGPT-small"
V4_DIR = "data_output_v4_lemma"
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 4 
MAX_LENGTH = 128 
MAX_NEW_TOKENS = 50 

# --- V3 EXPERIMENT: Increase Bipolar Weight Again ---
BIPOLAR_WEIGHT = 1.0 # <-- CAMBIO CLAVE: 100 veces más fuerte que el original
OUTPUT_DIR = "buss_lora_model_lambda_1_0" # <-- NUEVA CARPETA DE SALIDA
# -----------------------------------------------

# --- Define target scores based on p2_05 results ---
TARGET_POS_PROJECTION = -0.20 # 'great'
TARGET_NEG_PROJECTION = 0.30  # 'bad'

# --- CUDA Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Starting Pillar 3 (V3 - Lambda=1.0): BUSS-LORA Training ---")
print(f"Using device: {device}")
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. Training will be extremely slow on CPU.")

# ===================================================================
# STEP 1: Load BUSS Artifacts
# ===================================================================
print("Step 1: Loading BUSS artifacts (vectorizer and sentiment axis)...")

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
    
    sentiment_axis_vector = np.load(os.path.join(V4_DIR, "sentiment_axis_vector.npy"))
    sentiment_axis_vector = sentiment_axis_vector.reshape(-1, 1) # (5000, 1)

    print(f"BUSS artifacts loaded. Sentiment Axis Shape: {sentiment_axis_vector.shape}")

except FileNotFoundError as e:
    print(f"ERROR: Files not found in '{V4_DIR}'. {e}")
    exit()

# ===================================================================
# STEP 2: Define the Bipolar Projection Function
# ===================================================================
def get_buss_projection(text_batch):
    try:
        tfidf_vectors = vectorizer.transform(text_batch)
        projections = tfidf_vectors @ sentiment_axis_vector
        return torch.tensor(projections, dtype=torch.float32).to(device)
    except Exception as e:
        print(f"Error in get_buss_projection: {e}")
        return torch.zeros((len(text_batch), 1), dtype=torch.float32).to(device)

# ===================================================================
# STEP 3: Create Custom Dataset
# =================================S==================================
print("Step 3: Preparing IMDB Dataset...")

class SentimentDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_length=128):
        self.dataset = load_dataset("imdb", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label'] # 0 (neg) or 1 (pos)

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "sentiment_label": torch.tensor(label, dtype=torch.float32) 
        }

# ===================================================================
# STEP 4: Initialize Model, Tokenizer, and LoRA
# ===================================================================
print("Step 4: Initializing Model, Tokenizer, and LoRA...")

warnings.filterwarnings("ignore", message="A decoder-only architecture is being used")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)

model = get_peft_model(model, lora_config)
model.to(device) 
model.print_trainable_parameters() 

# ===================================================================
# STEP 5: Prepare DataLoader
# ===================================================================
train_dataset = SentimentDataset(tokenizer, split='train[:2000]') 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===================================================================
# STEP 6: Training Loop
# ===================================================================
print(f"\nStep 5: Starting Training Loop (Lambda = {BIPOLAR_WEIGHT})...")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = NUM_EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

model.train() 

for epoch in range(NUM_EPOCHS):
    print(f"\n--- EPOCH {epoch + 1} / {NUM_EPOCHS} ---")
    
    total_lm_loss = 0
    total_buss_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        sentiment_labels = batch["sentiment_label"].to(device) 
        
        optimizer.zero_grad()

        # --- 1. Pérdida de Lenguaje (LM Loss) ---
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        lm_loss = outputs.loss
        total_lm_loss += lm_loss.item()
        
        # --- 2. Pérdida Contrastiva Bipolar (BUSS Loss) ---
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_projections = get_buss_projection(generated_text) 

        # --- Create target projections based on labels ---
        target_projections = torch.full_like(generated_projections, TARGET_POS_PROJECTION)
        target_projections[sentiment_labels == 0] = TARGET_NEG_PROJECTION
        
        # --- Definición de la Pérdida (MSE) ---
        loss_fn = torch.nn.MSELoss()
        buss_loss = loss_fn(generated_projections, target_projections)
        
        total_buss_loss += buss_loss.item()

        # --- Combinar Pérdidas y Retropropagar ---
        total_loss = lm_loss + (buss_loss * BIPOLAR_WEIGHT) # <-- BIPOLAR_WEIGHT = 1.0
        
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

    # --- Fin de la Época ---
    avg_lm_loss = total_lm_loss / len(train_loader)
    avg_buss_loss = total_buss_loss / len(train_loader)
    print(f"Epoch {epoch+1} complete.")
    print(f"  Average LM Loss: {avg_lm_loss:.4f}")
    print(f"  Average BUSS Loss: {avg_buss_loss:.4f}") # <-- ¡La métrica clave!

# ===================================================================
# STEP 7: Save the Model
# ===================================================================
print("\nStep 6: Training complete. Saving model...")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")
print(f"\n--- Pillar 3 Training Complete (Lambda = {BIPOLAR_WEIGHT}) ---")