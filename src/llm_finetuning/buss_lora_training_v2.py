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
Versión v2.8: Prueba de Concepto Final. 
              Mantenemos la lógica v2.7 (que fue estable) pero 
              reemplazamos el DEMO_DATASET de 10 frases por uno de 50 
              para prevenir el "looping" por sobreajuste.
"""

# pip install torch transformers peft datasets accelerate scikit-learn numpy pypdf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings
import os
import glob
import re

warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATOS DE DEMOSTRACIÓN (v2 - Más Variedad)
# =============================================================================

# Usamos 50 frases únicas para prevenir el sobreajuste y el "looping"
DEMO_DATASET_V2 = [
    "Machine learning is the future of technology.",
    "Artificial intelligence can solve many world problems.",
    "The concept of this paper is novel and interesting.",
    "This new algorithm provides significant improvements.",
    "The data shows a positive trend.",
    "Bipolar scales capture semantic opposition.",
    "This method is highly efficient and scalable.",
    "We propose a new framework for data analysis.",
    "The results confirm our initial hypothesis.",
    "This model achieves state-of-the-art performance.",
    "Data privacy is a major concern in the digital age.",
    "Quantum computing will revolutionize complex simulations.",
    "The study of ethics in AI is crucial for development.",
    "Renewable energy is key to a sustainable future.",
    "Decentralized finance aims to disrupt traditional banking.",
    "Natural language processing allows computers to understand text.",
    "The historical context of this event is often overlooked.",
    "A new approach to drug discovery was presented.",
    "The correlation between these variables is statistically significant.",
    "We must consider the long-term impact of our decisions.",
    "This framework is robust against adversarial attacks.",
    "The neural network architecture consists of twelve layers.",
    "A decline in biodiversity threatens the ecosystem.",
    "The implementation of this policy had unintended consequences.",
    "This is a fundamental limitation of the current model.",
    "We identified a critical flaw in the existing methodology.",
    "The market showed high volatility this quarter.",
    "This research is purely theoretical and lacks practical application.",
    "The philosophical implications of this discovery are profound.",
    "This is a highly optimized and efficient solution.",
    "The dataset was small and inherently biased.",
    "This problem is computationally intractable.",
    "We must reject the null hypothesis.",
    "The model suffers from catastrophic forgetting.",
    "This paper presents a flawed and weak argument.",
    "The results are inconclusive and require further study.",
    "This method is outdated and inefficient.",
    "We argue against the current consensus.",
    "This is a minor improvement over the baseline.",
    "The theory is elegant but lacks empirical evidence.",
    "This is a complex and multifaceted issue.",
    "The algorithm fails to converge on this dataset.",
    "There is a fundamental misunderstanding of the core concept.",
    "This approach is simplistic and ignores key factors.",
    "The political situation is highly unstable.",
    "This company is facing financial ruin.",
    "Public opinion remains sharply divided on the matter.",
    "The treaty was a significant failure.",
    "This is a short-term solution to a long-term problem.",
    "The experiment was a complete success."
] * 4 # Multiplicamos para tener 200 ejemplos


# =============================================================================
# 2. CLASE BUSS (Modelo de Embeddings Bipolares)
# =============================================================================

class BUSS:
    """
    Implementa la Escala Semántica Universal Bipolar (BUSS) usando TF-IDF y SVD centrado.
    """
    def __init__(self, n_components=24, max_features=1000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        self.svd = TruncatedSVD(n_components=n_components)
        self.E_mean = None
        self.n_components = n_components

    def fit(self, texts):
        """
        Ajusta (entrena) el modelo BUSS en un corpus de textos.
        """
        try:
            E = self.vectorizer.fit_transform(texts).toarray()
            self.E_mean = E.mean(axis=0)
            E_c = E - self.E_mean
            
            n_samples, n_features = E_c.shape
            
            current_n_components = min(self.n_components, n_samples - 1, n_features - 1)
            if current_n_components < 1:
                current_n_components = 1

            if self.n_components != current_n_components:
                print(f"ADVERTENCIA: n_components ajustado a {current_n_components}")
                self.n_components = current_n_components
                self.svd = TruncatedSVD(n_components=self.n_components)

            self.svd.fit(E_c)
            print(f"BUSS fitted ({self.n_components} components): Bipolar opposites ready.")
        except Exception as e:
            print(f"Error durante el fit de BUSS: {e}")
            vocab_size = len(self.vectorizer.get_feature_names_out())
            if vocab_size == 0: vocab_size = self.max_features
            self.E_mean = np.zeros(vocab_size)


    def project(self, new_texts):
        """
        Proyecta nuevos textos en el espacio bipolar aprendido.
        """
        if self.E_mean is None:
            raise RuntimeError("El modelo BUSS debe ser entrenado (fit) antes de proyectar.")
        
        try:
            E_new = self.vectorizer.transform(new_texts).toarray()
            E_new_c = E_new - self.E_mean
            P_new = self.svd.transform(E_new_c)
            
            norm = np.linalg.norm(P_new, axis=1, keepdims=True)
            P_norm = np.where(norm > 1e-6, P_new / norm, 0.0)
            return P_norm
        except Exception as e:
            print(f"Error al proyectar textos: {e}")
            return np.zeros((len(new_texts), self.n_components))

    def opposite(self, P_norm):
        """
        Calcula el opuesto bipolar perfecto (Teorema de Oposición Perfecta).
        v(-C) = -v(C)
        """
        return -P_norm

# =============================================================================
# 3. DATASET Y COLLATOR BIPOLARES
# =============================================================================

class BipolarDataset(Dataset):
    """
    Crea pares de datos: (Texto Original, Texto Opuesto Simulado).
    """
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [t for t in texts if t]
        
        self.opp_texts = [f"Bipolar_Opposite: {t}" for t in self.texts]

    def __len__(self):
        return len(self.texts) * 2

    def __getitem__(self, idx):
        is_opposite = 0
        if idx < len(self.texts):
            text = self.texts[idx]
        else:
            text = self.opp_texts[idx - len(self.texts)]
            is_opposite = 1
            
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
    Collator personalizado que maneja 'is_opposite'.
    """
    def __call__(self, features):
        is_opposite_labels = [f.pop('is_opposite') for f in features]
        batch = super().__call__(features)
        batch['is_opposite'] = torch.stack(is_opposite_labels)
        return batch

# =============================================================================
# 4. PÉRDIDA Y TRAINER BIPOLARES (ESTABILIZADO v2.4)
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
            # Usamos Tanh para "aplastar" los logits a un rango [-1, 1].
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
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        loss = self.loss_function(logits, labels, is_opposite)
        
        return (loss, outputs) if return_outputs else loss

# =============================================================================
# 5. PIPELINE DE ENTRENAMIENTO PRINCIPAL
# =============================================================================

def train_buss_lora():
    
    # --- Parámetros de Entrenamiento Estables (v2.6) ---
    N_EPOCHS = 15
    BIPOLAR_WEIGHT = 0.001 
    OUTPUT_DIR = "./buss_lora_final_v2"
    MODEL_NAME = "microsoft/DialoGPT-small"
    LEARNING_RATE = 5e-5
    
    # --- 1. Cargar Datos de DEMO v2 ---
    print(f"--- 1. Cargando datos de DEMO_DATASET_V2 ---")
    texts = DEMO_DATASET_V2
    print(f"Se cargaron {len(texts)} ejemplos de demostración.")

    # --- 2. Inicializar y Entrenar BUSS ---
    print("🚀 Initializing BUSS...")
    # n_components debe ser < n_samples (200)
    n_components = min(24, len(texts) - 1) 
    buss = BUSS(n_components=n_components)
    buss.fit(texts) 

    # --- 3. Cargar Modelo y Tokenizer ---
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # --- 4. Configurar LoRA (PEFT) ---
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

    # --- 5. Preparar Dataset y Collator Bipolares ---
    dataset = BipolarDataset(texts, tokenizer, max_length=64) # Max_length más corto para demo
    data_collator = BipolarDataCollator(tokenizer=tokenizer, mlm=False)

    # --- 6. Inicializar nuestra Pérdida Bipolar ---
    bipolar_loss_func = BipolarLoss(bipolar_weight=BIPOLAR_WEIGHT)

    # --- 7. Configurar Argumentos de Entrenamiento ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=2,
        warmup_steps=20,
        logging_steps=10,
        save_steps=50,
        report_to="none",
        learning_rate=LEARNING_RATE,
        fp16=False,
        max_grad_norm=1.0 
    )

    # --- 8. Inicializar nuestro Trainer Bipolar ---
    trainer = BipolarTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        loss_function=bipolar_loss_func
    )

    # --- 9. Entrenar ---
    print(f"🧠 Training BUSS-LoRA (Epochs: {N_EPOCHS}, Bipolar Weight: {BIPOLAR_WEIGHT}, LR: {LEARNING_RATE})...")
    trainer.train()

    # --- 10. Guardar el modelo final ---
    print(f"✅ Training complete. Saving model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved successfully.")

    # --- 11. Test rápido de generación ---
    print("\n--- TEST: GENERATION ---")
    prompt_std = "This model achieves" # Un prompt positivo
    prompt_neg = "This method is" # Un prompt negativo/crítico
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # --- Test 1 (Positivo) ---
    print(f"\n--- Test 1 (Prompt Positivo) ---")
    inputs_std = tokenizer(prompt_std, return_tensors="pt").to(device)
    outputs_std = model.generate(**inputs_std, max_length=30, num_beams=5, no_repeat_ngram_size=2) 
    print(f"Generated (Standard):")
    print(f"   {tokenizer.decode(outputs_std[0], skip_special_tokens=True)}")

    prompt_opp_1 = f"Bipolar_Opposite: {prompt_std}"
    inputs_opp_1 = tokenizer(prompt_opp_1, return_tensors="pt").to(device)
    outputs_opp_1 = model.generate(**inputs_opp_1, max_length=30, num_beams=5, no_repeat_ngram_size=2)
    print(f"Generated (Bipolar_Opposite):")
    print(f"   {tokenizer.decode(outputs_opp_1[0], skip_special_tokens=True)}")

    # --- Test 2 (Negativo) ---
    print(f"\n--- Test 2 (Prompt Negativo/Crítico) ---")
    inputs_neg = tokenizer(prompt_neg, return_tensors="pt").to(device)
    outputs_neg = model.generate(**inputs_neg, max_length=30, num_beams=5, no_repeat_ngram_size=2) 
    print(f"Generated (Standard):")
    print(f"   {tokenizer.decode(outputs_neg[0], skip_special_tokens=True)}")
    
    prompt_opp_2 = f"Bipolar_Opposite: {prompt_neg}"
    inputs_opp_2 = tokenizer(prompt_opp_2, return_tensors="pt").to(device)
    outputs_opp_2 = model.generate(**inputs_opp_2, max_length=30, num_beams=5, no_repeat_ngram_size=2)
    print(f"Generated (Bipolar_Opposite):")
    print(f"   {tokenizer.decode(outputs_opp_2[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    train_buss_lora()

