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
BUSS + LoRA: Bipolar Universal Semantic Scale for LLM Fine-Tuning
Autor: J. Arturo Ornelas Brand
Versión v5.0: ¡Versión de Producción/Prueba Real!

Este script implementa el 'approach' estable v4.0 (prefijo + Tanh Loss)
pero reemplaza el PAIRED_DATASET de demostración con el
dataset real IMDB Movie Review (cargado desde la carpeta 'aclImdb').

Esto entrena al modelo en miles de ejemplos de "ground truth"
bipolares (positivos vs. negativos).
"""

# pip install torch transformers peft datasets accelerate scikit-learn numpy
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import warnings
import os
import glob
import re

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CARGADOR DE DATOS (IMDB REAL)
# =============================================================================

def load_imdb_data(base_dir, max_samples_per_class=1000):
    """
    Carga los archivos .txt de las carpetas train/pos y train/neg
    del dataset aclImdb.
    """
    texts_standard = []
    texts_opposite = []
    
    pos_dir = os.path.join(base_dir, "train", "pos")
    neg_dir = os.path.join(base_dir, "train", "neg")

    print(f"--- 1. Cargando datos reales desde {base_dir} ---")
    
    # Cargar Textos Positivos (Standard)
    pos_files = glob.glob(os.path.join(pos_dir, "*.txt"))
    for file_path in pos_files[:max_samples_per_class]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts_standard.append(f.read())
        except Exception as e:
            print(f"Advertencia: No se pudo leer {file_path}: {e}")

    # Cargar Textos Negativos (Opposite)
    neg_files = glob.glob(os.path.join(neg_dir, "*.txt"))
    for file_path in neg_files[:max_samples_per_class]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Añadimos el prefijo al texto opuesto real
                texts_opposite.append(f"Bipolar_Opposite: {f.read()}")
        except Exception as e:
            print(f"Advertencia: No se pudo leer {file_path}: {e}")

    if not texts_standard or not texts_opposite:
        print(f"Error: No se encontraron archivos de texto en {pos_dir} o {neg_dir}.")
        print("¿Descargaste y descomprimiste 'aclImdb_v1.tar.gz' en esta carpeta?")
        return [], []

    print(f"Se cargaron {len(texts_standard)} críticas positivas (Standard).")
    print(f"Se cargaron {len(texts_opposite)} críticas negativas (Bipolar_Opposite).")
    
    return texts_standard, texts_opposite

# =============================================================================
# 2. DATASET Y COLLATOR BIPOLARES (v4.0 - ESTABLE)
# =============================================================================

class BipolarDataset(Dataset):
    """
    Toma dos listas (standard, opposite) y las presenta como una
    lista única con una bandera 'is_opposite'.
    """
    def __init__(self, texts_standard, texts_opposite, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.texts_standard = texts_standard
        self.texts_opposite = texts_opposite
        
        # Combinamos todo en una sola lista
        self.all_texts = self.texts_standard + self.texts_opposite
        self.len_standard = len(self.texts_standard)
        print(f"BipolarDataset inicializado: {self.len_standard} Standard, {len(self.texts_opposite)} Opposite.")

    def __len__(self):
        return len(self.all_texts)

    def __getitem__(self, idx):
        text = self.all_texts[idx]
        is_opposite = 1 if idx >= self.len_standard else 0
            
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding['labels'] = encoding['input_ids'].clone().squeeze(0)
        encoding['input_ids'] = encoding['input_ids'].squeeze(0)
        encoding['attention_mask'] = encoding['attention_mask'].squeeze(0)
        encoding['is_opposite'] = torch.tensor(is_opposite, dtype=torch.long)
        
        return encoding

class BipolarDataCollator(DataCollatorForLanguageModeling):
    """
    Collator personalizado que hereda de la base Y maneja 'is_opposite'.
    """
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)

    def __call__(self, features):
        is_opposite_labels = [f.pop('is_opposite') for f in features]
        batch = super().__call__(features)
        batch['is_opposite'] = torch.stack(is_opposite_labels)
        return batch

# =============================================================================
# 3. PÉRDIDA Y TRAINER BIPOLARES (Estable, con Tanh)
# =============================================================================

class BipolarLoss(torch.nn.Module):
    """
    Calcula CE y añade una penalización de contraste bipolar ESTABLE.
    """
    def __init__(self, bipolar_weight=1.0):
        super().__init__()
        self.bipolar_weight = bipolar_weight
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels, is_opposite):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_ce = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), 
                               shift_labels.view(-1))
        
        bipolar_penalty = 0.0
        opposite_logits = logits[is_opposite == 1]
        
        if opposite_logits.numel() > 0:
            stable_logits = torch.tanh(opposite_logits)
            bipolar_penalty = torch.mean(stable_logits.pow(2)) * self.bipolar_weight

        total_loss = loss_ce + bipolar_penalty
        return total_loss

class BipolarTrainer(Trainer):
    """
    Trainer personalizado que maneja 'is_opposite' y la BipolarLoss.
    """
    def __init__(self, *args, loss_function, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        is_opposite = inputs.pop("is_opposite")
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = self.loss_function(logits, labels, is_opposite)
        
        return (loss, outputs) if return_outputs else loss

# =============================================================================
# 4. PIPELINE DE ENTRENAMIENTO PRINCIPAL
# =============================================================================

def train_buss_lora_v5():
    
    # --- Parámetros de Entrenamiento ---
    # Reducimos épocas porque el dataset es MUCHO más grande
    N_EPOCHS = 3 
    BIPOLAR_WEIGHT = 0.001 # Mantenemos el peso sutil
    OUTPUT_DIR = "./buss_lora_final_v5_imdb"
    MODEL_NAME = "microsoft/DialoGPT-small"
    LEARNING_RATE = 5e-5
    # Limitar el número de muestras para un entrenamiento de prueba rápido
    # Poner en -1 para usar los 25,000
    MAX_SAMPLES_PER_CLASS = 2000 # Usaremos 2000 positivos + 2000 negativos = 4000 total
    
    # --- 1. Cargar Datos Reales de IMDB ---
    texts_standard, texts_opposite = load_imdb_data(
        base_dir="aclImdb", 
        max_samples_per_class=MAX_SAMPLES_PER_CLASS
    )
    if not texts_standard:
        return

    # --- 2. Cargar Modelo y Tokenizer ---
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # --- 3. Configurar LoRA (PEFT) ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    print("LoRA model configured.")
    model.print_trainable_parameters()

    # --- 4. Preparar Dataset y Collator Bipolares ---
    dataset = BipolarDataset(texts_standard, texts_opposite, tokenizer, max_length=128) # max_length de 128
    data_collator = BipolarDataCollator(tokenizer=tokenizer, mlm=False)

    # --- 5. Inicializar nuestra Pérdida Bipolar ---
    bipolar_loss_func = BipolarLoss(bipolar_weight=BIPOLAR_WEIGHT)

    # --- 6. Configurar Argumentos de Entrenamiento ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4, # Acumulación (4*4=16 batch efectivo)
        warmup_steps=100,
        logging_steps=50,
        save_steps=200,
        report_to="none",
        learning_rate=LEARNING_RATE,
        fp16=False,
        max_grad_norm=1.0 
    )

    # --- 7. Inicializar nuestro Trainer Bipolar ---
    trainer = BipolarTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        loss_function=bipolar_loss_func
    )

    # --- 8. Entrenar ---
    print(f"🧠 Training BUSS-LoRA v5 (IMDB) (Epochs: {N_EPOCHS}, Bipolar Weight: {BIPOLAR_WEIGHT}, LR: {LEARNING_RATE})...")
    trainer.train()

    # --- 9. Guardar el modelo final ---
    print(f"✅ Training complete. Saving model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved successfully.")

    # --- 10. Test rápido de generación ---
    print("\n--- TEST: GENERATION (v5) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Probamos generar desde un prompt POSITIVO
    prompt_anchor = "This movie was absolutely fantastic"
    inputs_anchor = tokenizer(prompt_anchor, return_tensors="pt").to(device)
    outputs_anchor = model.generate(**inputs_anchor, max_length=40, num_beams=5, no_repeat_ngram_size=2) 
    print(f"Generated (Anchor Prompt - Positive):")
    print(f"   {tokenizer.decode(outputs_anchor[0], skip_special_tokens=True)}")

    # Probamos generar desde un prompt NEGATIVO (usando el prefijo)
    prompt_opposite = "Bipolar_Opposite: This movie was absolutely terrible"
    inputs_opp = tokenizer(prompt_opposite, return_tensors="pt").to(device)
    outputs_opp = model.generate(**inputs_opp, max_length=40, num_beams=5, no_repeat_ngram_size=2)
    print(f"Generated (Opposite Prompt - Negative):")
    print(f"   {tokenizer.decode(outputs_opp[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    train_buss_lora_v5()

