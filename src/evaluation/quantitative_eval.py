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
Author: J. Arturo Ornelas Brand
Integrates BUSS bipolar embeddings into LoRA training with Bipolar Loss.
Generates opposites via SVD flip for robust training.
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
from datasets import Dataset as HFDataset # Renombrado para evitar conflicto
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. BUSS EMBEDDING CLASS (Core SVD Bipolar)
# =============================================================================

class BUSS:
    def __init__(self, texts, n_components=64):
        # Aseguramos que haya suficientes componentes si el dataset es pequeño
        N = len(texts)
        self.n_components = min(n_components, N - 1) 
        if self.n_components <= 0:
             raise ValueError("Necesitas más documentos que componentes SVD.")

        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.svd = TruncatedSVD(n_components=self.n_components)
        self.texts = texts
        self.fit()
    
    def fit(self):
        E = self.vectorizer.fit_transform(self.texts).toarray()
        E_c = E - E.mean(axis=0)
        self.P = self.svd.fit_transform(E_c)
        self.P_norm = self.P / np.linalg.norm(self.P, axis=1, keepdims=True)
    
    def project(self, new_texts):
        E_new = self.vectorizer.transform(new_texts).toarray()
        E_new_c = E_new - self.vectorizer.transform(self.texts).toarray().mean(axis=0)
        P_new = self.svd.transform(E_new_c)
        return P_new / np.linalg.norm(P_new, axis=1, keepdims=True)
    
    def opposite(self, P):
        """Bipolar flip: v(-C) = -v(C)"""
        return -P

# =============================================================================
# 2. BIPOLAR DATASET (Augment with Opposites and Bipolar Label)
# =============================================================================

class BipolarDataset(Dataset):
    """
    Combina textos originales y textos opuestos simulados.
    Añade la clave 'is_opposite' al diccionario de retorno.
    """
    def __init__(self, texts, tokenizer, buss, max_length=512):
        self.tokenizer = tokenizer
        self.buss = buss
        self.max_length = max_length
        
        # Textos originales
        self.texts = texts
        
        # Generar opuestos (usando placeholder para simular texto opuesto)
        embeddings = buss.project(texts)
        opp_embeddings = buss.opposite(embeddings)
        
        # Simulación del texto opuesto. En una aplicación real,
        # esto sería el texto semánticamente opuesto al original.
        # Aquí usamos un prefijo para que el modelo aprenda a asociar
        # el prefijo con un 'contexto opuesto'.
        self.opp_texts = [f"Bipolar_Opposite: {t}" for t in self.texts] 
    
    def __len__(self):
        return len(self.texts) * 2  # Original + Opuesto
    
    def __getitem__(self, idx):
        if idx < len(self.texts):
            # Caso 1: Texto Original (is_opposite = 0)
            text = self.texts[idx]
            is_opposite = torch.tensor(0, dtype=torch.long)
        else:
            # Caso 2: Texto Opuesto (is_opposite = 1)
            text = self.opp_texts[idx - len(self.texts)]
            is_opposite = torch.tensor(1, dtype=torch.long)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Ajustamos las etiquetas para ser input_ids
        encoding['labels'] = encoding['input_ids'].clone().squeeze(0)
        
        # Añadimos la bandera bipolar, que el DataCollator debe manejar
        encoding['is_opposite'] = is_opposite
        
        # El Trainer espera solo tensores, así que devolvemos un diccionario simple de tensores
        return {k: v.squeeze(0) if v.dim() > 1 else v for k, v in encoding.items()}

# =============================================================================
# 3. BIPOLAR LOSS (La Innovación Conceptual)
# =============================================================================

class BipolarLoss(torch.nn.Module):
    """
    Calcula la Pérdida de Entropía Cruzada (CE) estándar 
    y añade una penalización conceptual para el caso 'opuesto'.
    """
    def __init__(self, bipolar_weight=0.1):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.bipolar_weight = bipolar_weight
    
    def forward(self, logits, labels, is_opposite=None):
        
        # 1. Pérdida Estándar (Language Modeling Loss)
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tensors for CrossEntropyLoss
        ce_loss = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 2. Pérdida Bipolar Conceptual (Simulación de Contraste)
        # --------------------------------------------------------
        bipolar_penalty = 0.0

        if is_opposite is not None:
            # Identificar los índices de los ejemplos que son "opuestos"
            # (El DataCollator debe haber apilado 'is_opposite' correctamente)
            
            # Nota: Esto es una simplificación. En un modelo real, aquí se
            # calcularía la distancia vectorial entre el embedding de la secuencia
            # y el vector BUSS esperado (P vs. -P).
            
            # Usaremos una simple penalización si la predicción es trivial
            # para forzar al modelo a tener que procesar la dualidad.
            
            is_opposite_mask = is_opposite.bool()
            
            if is_opposite_mask.any():
                # En un entrenamiento real, si el modelo está en modo 'opuesto',
                # y genera un texto que se proyecta en la misma dirección que el
                # original (es decir, P(LLM_Output) ~ P_original), debe ser penalizado.
                
                # Aquí, la penalización es un término L2 simple sobre los logits
                # de las secuencias opuestas, forzando la dispersión.
                
                # Tomamos los logits de las secuencias marcadas como opuestas
                opp_logits = logits[is_opposite_mask]
                
                # Penalización: L2 de los logits de las secuencias opuestas.
                # Esto es heurístico, forzando al modelo a 'pensar diferente'
                # en el contexto "Bipolar_Opposite:", lo que evita la
                # trivialidad de replicar el patrón del texto original.
                bipolar_penalty = torch.norm(opp_logits, p=2) / opp_logits.numel()
                
        # Pérdida Total
        total_loss = ce_loss + self.bipolar_weight * bipolar_penalty
        return total_loss

# =============================================================================
# 4. CUSTOM TRAINER (Para manejar la pérdida con el parámetro extra)
# =============================================================================

class BipolarTrainer(Trainer):
    # ! IMPORTANTE: Añadimos __init__ para manejar el argumento custom 'loss_function'
    def __init__(self, loss_function, *args, **kwargs):
        # 1. Almacenamos la función de pérdida custom
        self.loss_function = loss_function
        # 2. Pasamos todos los demás argumentos al constructor de la clase base Trainer
        super().__init__(*args, **kwargs) 

    # ! CORRECCIÓN: Añadir 'num_items_in_batch' para compatibilidad con versiones antiguas
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extraer el tensor 'is_opposite' que contiene la etiqueta bipolar
        is_opposite = inputs.pop("is_opposite", None) 
        
        # Forward pass estándar
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        # Usar la BipolarLoss, pasando el parámetro extra
        loss = self.loss_function(logits, labels, is_opposite=is_opposite)

        return (loss, outputs) if return_outputs else loss

# =============================================================================
# 5. DATA COLLATOR (Para manejar la etiqueta bipolar en lotes)
# =============================================================================

class BipolarDataCollator(DataCollatorForLanguageModeling):
    """
    Extiende el DataCollator estándar para manejar y apilar 
    el tensor 'is_opposite' y eliminarlo de los 'labels'
    que se pasan al modelo directamente.
    """
    def torch_call(self, examples):
        # Separar el tensor 'is_opposite' si existe
        if "is_opposite" in examples[0]:
            is_opposite_list = [ex.pop("is_opposite") for ex in examples]
            is_opposite = torch.stack(is_opposite_list)
        else:
            is_opposite = None
            
        # Llamar al DataCollator original
        batch = super().torch_call(examples)
        
        # Reincorporar la etiqueta bipolar al lote
        if is_opposite is not None:
            batch['is_opposite'] = is_opposite
            
        return batch

# =============================================================================
# 6. MAIN TRAINING PIPELINE
# =============================================================================

def train_buss_lora():
    # Demo dataset (replace with your data)
    texts = [
        "Machine learning uses embeddings for semantic similarity.",
        "Neural networks learn hierarchical representations.",
        "SVD decomposes matrices into orthogonal components.",
        "Bipolar scales capture opposition in language.",
        "LoRA enables efficient fine-tuning of LLMs."
    ] * 5  # 25 examples

    # 1. BUSS Embeddings
    print("🚀 Initializing BUSS...")
    try:
        buss = BUSS(texts)
    except ValueError as e:
        print(f"Error BUSS: {e}")
        return
    print(f"BUSS fitted ({buss.n_components} components): Bipolar opposites ready.")
    
    # 2. Load Model
    model_name = "microsoft/DialoGPT-small"  # Small for demo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 3. LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Adaptar target_modules a DialoGPT-small (basado en GPT-2)
        target_modules=["c_attn", "c_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())
    
    # 4. Bipolar Dataset
    dataset = BipolarDataset(texts, tokenizer, buss)
    
    # 5. Data Collator and Loss Function
    data_collator = BipolarDataCollator(tokenizer, mlm=False)
    bipolar_loss_function = BipolarLoss(bipolar_weight=0.5) # Aumentamos el peso
    
    # 6. Training Args
    training_args = TrainingArguments(
        output_dir="./buss_lora_model",
        num_train_epochs=5,
        per_device_train_batch_size=2, # Reducido para simulación
        gradient_accumulation_steps=1, 
        warmup_steps=10,
        logging_steps=5,
        save_steps=100,
        report_to="none" # Eliminado 'evaluation_strategy' para compatibilidad con versiones antiguas
    )
    
    # 7. Bipolar Trainer (Usamos el custom Trainer)
    trainer = BipolarTrainer(
        # Argumento custom manejado en el __init__ de BipolarTrainer
        loss_function=bipolar_loss_function, 
        # Argumentos estándar pasados a la clase base Trainer
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 8. Train
    print("\n🧠 Training BUSS-LoRA with Bipolar Contrastive Loss...")
    trainer.train()
    
    # 9. Save
    trainer.save_model("./buss_lora_final")
    tokenizer.save_pretrained("./buss_lora_final")
    print("\n✅ Model saved: buss_lora_final/")
    
    # 10. Test
    test_text = "Bipolar scales capture opposition"
    opp_text = "Bipolar_Opposite: Bipolar scales capture opposition"
    
    print("\n--- TEST: GENERATION ---")
    
    # Generación Estándar (Sin prefijo)
    inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    print("\nGenerated (Standard, expected similarity):")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    # Generación Opuesta (Con prefijo)
    inputs_opp = tokenizer(opp_text, return_tensors="pt").to(model.device)
    outputs_opp = model.generate(**inputs_opp, max_length=50, num_return_sequences=1)
    print("\nGenerated (Bipolar_Opposite, expected contrast/dispersion):")
    print(tokenizer.decode(outputs_opp[0], skip_special_tokens=True))

if __name__ == "__main__":
    train_buss_lora()
