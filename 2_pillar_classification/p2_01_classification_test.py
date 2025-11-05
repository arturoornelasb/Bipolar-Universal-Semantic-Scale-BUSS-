# Copyright 2025 José Arturo Ornelas Brand

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from datasets import load_dataset
import time
import os

print("--- Starting Pillar 2: Classification Test ---")

# --- Configuration ---
# We use the V3 data (100 components), our winner from Pillar 1
V3_DATA_DIR = "data_output"
N_JOBS = -1 # Use all CPU cores for cross-validation
CV_FOLDS = 5  # 5-Fold cross-validation as recommended

# ===================================================================
# STEP 1: Load Labels (y)
# ===================================================================
print("Step 1: Loading labels (y) from Hugging Face dataset...")
start_time = time.time()
imdb_train = load_dataset("imdb", split="train")
imdb_test = load_dataset("imdb", split="test")
y = np.array(list(imdb_train['label']) + list(imdb_test['label']))
print(f"Labels loaded in {time.time() - start_time:.2f}s. Shape: {y.shape}")

# ===================================================================
# STEP 2: Load Matrices (X)
# ===================================================================
print("Step 2: Loading matrices E, V_buss, and V_lsa...")
try:
    # 1. Full TF-IDF (Our "Ceiling" model)
    X_tfidf = load_npz(os.path.join(V3_DATA_DIR, "E.npz"))
    
    # 2. BUSS Projections
    V_buss = np.load(os.path.join(V3_DATA_DIR, "V_buss.npy"))
    X_buss = X_tfidf @ V_buss # (50000, 5000) @ (5000, 100) -> (50000, 100)
    
    # 3. LSA Projections
    V_lsa = np.load(os.path.join(V3_DATA_DIR, "V_lsa.npy"))
    X_lsa = X_tfidf @ V_lsa # (50000, 100)

except FileNotFoundError:
    print(f"ERROR: Files not found in '{V3_DATA_DIR}'.")
    print("Please ensure the V3 data (E.npz, V_buss.npy, V_lsa.npy) exists.")
    exit()

print("All features (X) loaded and prepared:")
print(f"  X_tfidf (Ceiling): {X_tfidf.shape}")
print(f"  X_buss (100-dim):  {X_buss.shape}")
print(f"  X_lsa (100-dim):   {X_lsa.shape}")

# ===================================================================
# STEP 3: Run Cross-Validation
# ===================================================================
print(f"\nStep 3: Running {CV_FOLDS}-Fold Cross-Validation...")

# We use a simple, strong Logistic Regression model for all tests
# max_iter=1000 ensures convergence. n_jobs=1 inside LR to avoid conflicts.
model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)

# --- Model 1: TF-IDF (Ceiling) ---
print(f"  Testing Model 1 (TF-IDF Ceiling)... (This will take a few minutes)")
start_time = time.time()
scores_tfidf = cross_val_score(model, X_tfidf, y, cv=CV_FOLDS, n_jobs=N_JOBS, scoring='accuracy')
print(f"  ...done in {time.time() - start_time:.2f}s")

# --- Model 2: BUSS (100 Features) ---
print(f"  Testing Model 2 (BUSS 100-dim)...")
start_time = time.time()
scores_buss = cross_val_score(model, X_buss, y, cv=CV_FOLDS, n_jobs=N_JOBS, scoring='accuracy')
print(f"  ...done in {time.time() - start_time:.2f}s")

# --- Model 3: LSA (100 Features) ---
print(f"  Testing Model 3 (LSA 100-dim)...")
start_time = time.time()
scores_lsa = cross_val_score(model, X_lsa, y, cv=CV_FOLDS, n_jobs=N_JOBS, scoring='accuracy')
print(f"  ...done in {time.time() - start_time:.2f}s")

# ===================================================================
# STEP 4: Print Results
# ===================================================================
print("\n--- Pillar 2 Results: Classification Accuracy (Mean ± Std) ---")
print(f"  1. TF-IDF (Ceiling):   {np.mean(scores_tfidf) * 100:.2f}% (± {np.std(scores_tfidf) * 100:.2f})")
print(f"  2. BUSS (100-dim):     {np.mean(scores_buss) * 100:.2f}% (± {np.std(scores_buss) * 100:.2f})")
print(f"  3. LSA (100-dim):      {np.mean(scores_lsa) * 100:.2f}% (± {np.std(scores_lsa) * 100:.2f})")

print("\n--- Verdict ---")
acc_buss = np.mean(scores_buss)
acc_lsa = np.mean(scores_lsa)

if acc_buss > acc_lsa:
    print(f"SUCCESS: BUSS ({acc_buss*100:.2f}%) is more accurate than LSA ({acc_lsa*100:.2f}%).")
    print(f"BUSS retained {acc_buss / np.mean(scores_tfidf) * 100:.1f}% of the full model's performance.")
else:
    print(f"FAILURE: LSA ({acc_lsa*100:.2f}%) was more accurate than BUSS ({acc_buss*100:.2f}%).")

print("\n--- Pillar 2 (Part 1) Complete ---")