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

print("--- Starting Pillar 2 (Part 2): V3 vs V4 Accuracy Test ---")

# --- Configuration ---
V3_DIR = "data_output"
V4_DIR = "data_output_v4_lemma"
N_JOBS = -1 # Use all CPU cores
CV_FOLDS = 5 

# ===================================================================
# STEP 1: Load Labels (y)
# ===================================================================
print("Step 1: Loading labels (y)...")
start_time = time.time()
imdb_train = load_dataset("imdb", split="train")
imdb_test = load_dataset("imdb", split="test")
y = np.array(list(imdb_train['label']) + list(imdb_test['label']))
print(f"Labels loaded in {time.time() - start_time:.2f}s. Shape: {y.shape}")

# ===================================================================
# STEP 2: Load V3 and V4 Data
# ===================================================================
print("Step 2: Loading all V3 (100-dim) and V4 (500-dim) data...")
try:
    # --- V3 Data (100 Components) ---
    X_tfidf_v3 = load_npz(os.path.join(V3_DIR, "E.npz"))
    V_buss_v3 = np.load(os.path.join(V3_DIR, "V_buss.npy"))
    V_lsa_v3 = np.load(os.path.join(V3_DIR, "V_lsa.npy"))
    # Create V3 projection features
    X_buss_v3 = X_tfidf_v3 @ V_buss_v3
    X_lsa_v3 = X_tfidf_v3 @ V_lsa_v3
    print("V3 (100-dim) features created.")

    # --- V4 Data (500 Components, Lemmatized) ---
    X_tfidf_v4 = load_npz(os.path.join(V4_DIR, "E_lemma.npz"))
    V_buss_v4 = np.load(os.path.join(V4_DIR, "V_buss_lemma.npy"))
    V_lsa_v4 = np.load(os.path.join(V4_DIR, "V_lsa_lemma.npy"))
    # Create V4 projection features
    X_buss_v4 = X_tfidf_v4 @ V_buss_v4
    X_lsa_v4 = X_tfidf_v4 @ V_lsa_v4
    print("V4 (500-dim) features created.")

except FileNotFoundError as e:
    print(f"ERROR: Files not found. {e}")
    exit()

# ===================================================================
# STEP 3: Run Cross-Validation on all models
# ===================================================================
print(f"\nStep 3: Running {CV_FOLDS}-Fold Cross-Validation on all 4 models...")
model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)

# --- V3 Models ---
print("  Testing V3 (100-dim)...")
scores_buss_v3 = cross_val_score(model, X_buss_v3, y, cv=CV_FOLDS, n_jobs=N_JOBS, scoring='accuracy')
scores_lsa_v3 = cross_val_score(model, X_lsa_v3, y, cv=CV_FOLDS, n_jobs=N_JOBS, scoring='accuracy')

# --- V4 Models ---
print("  Testing V4 (500-dim)...")
scores_buss_v4 = cross_val_score(model, X_buss_v4, y, cv=CV_FOLDS, n_jobs=N_JOBS, scoring='accuracy')
scores_lsa_v4 = cross_val_score(model, X_lsa_v4, y, cv=CV_FOLDS, n_jobs=N_JOBS, scoring='accuracy')

print("...All models tested.")

# ===================================================================
# STEP 4: Print Results
# ===================================================================
print("\n--- Pillar 2 Results: V3 (100-dim) vs V4 (500-dim) ---")
print("                          |   Accuracy (Mean)   |   Std Dev")
print("--------------------------|---------------------|------------")
print(f" V3 BUSS (100-dim)        |   {np.mean(scores_buss_v3) * 100:<17.2f} |   {np.std(scores_buss_v3) * 100:.2f}")
print(f" V3 LSA (100-dim)         |   {np.mean(scores_lsa_v3) * 100:<17.2f} |   {np.std(scores_lsa_v3) * 100:.2f}")
print(f" V4 BUSS (500-dim)        |   {np.mean(scores_buss_v4) * 100:<17.2f} |   {np.std(scores_buss_v4) * 100:.2f}")
print(f" V4 LSA (500-dim)         |   {np.mean(scores_lsa_v4) * 100:<17.2f} |   {np.std(scores_lsa_v4) * 100:.2f}")


print("\n--- Verdict ---")
acc_v3 = np.mean(scores_buss_v3)
acc_v4 = np.mean(scores_buss_v4)

if acc_v3 > acc_v4:
    print(f"SUCCESS: V3 (100-dim) @ {acc_v3*100:.2f}% is more accurate than V4 (500-dim) @ {acc_v4*100:.2f}%.")
    print("This confirms our hypothesis: the extra 400 components were 'topical noise' that HURT performance.")
else:
    print(f"FINDING: V4 (500-dim) @ {acc_v4*100:.2f}% was more accurate than V3 (100-dim) @ {acc_v3*100:.2f}%.")
    print("This means the extra components provided more useful information, not just noise.")

print("\n--- Pillar 2 Complete ---")