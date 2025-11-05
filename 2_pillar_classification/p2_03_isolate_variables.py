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
from scipy.stats import ttest_ind # <-- NEW: For statistical test
import time
import os

print("--- Starting Pillar 2 (Part 3): Isolating Variables Test ---")

# --- Configuration ---
V3_DIR = "data_output"
V4_DIR = "data_output_v4_lemma"
N_JOBS = -1 
CV_FOLDS = 5 

# ===================================================================
# STEP 1: Load Labels (y)
# ===================================================================
print("Step 1: Loading labels (y)...")
start_time = time.time()
imdb_train = load_dataset("imdb", split="train")
imdb_test = load_dataset("imdb", split="test")
y = np.array(list(imdb_train['label']) + list(imdb_test['label']))
print(f"Labels loaded in {time.time() - start_time:.2f}s.")

# ===================================================================
# STEP 2: Prepare 3 Feature Sets (X)
# ===================================================================
print("Step 2: Preparing 3 feature sets for comparison...")
try:
    # --- Model 1: V3 (Stopwords, 100-dim) ---
    X_tfidf_v3 = load_npz(os.path.join(V3_DIR, "E.npz"))
    V_buss_v3 = np.load(os.path.join(V3_DIR, "V_buss.npy"))
    X_buss_v3 = X_tfidf_v3 @ V_buss_v3
    print("  Loaded Model 1: V3 (Stopwords, 100-dim)")

    # --- Load V4 Data (Lemma, 500-dim) ---
    X_tfidf_v4 = load_npz(os.path.join(V4_DIR, "E_lemma.npz"))
    V_buss_v4_full = np.load(os.path.join(V4_DIR, "V_buss_lemma.npy")) # (5000, 500)
    
    # --- Model 2: V4 (Lemma, 100-dim) ---
    # We slice the V4 axes to get only the first 100 components
    V_buss_v4_100 = V_buss_v4_full[:, :100]
    X_buss_v4_100 = X_tfidf_v4 @ V_buss_v4_100
    print("  Loaded Model 2: V4 (Lemma, 100-dim)")

    # --- Model 3: V4 (Lemma, 500-dim) ---
    X_buss_v4_500 = X_tfidf_v4 @ V_buss_v4_full
    print("  Loaded Model 3: V4 (Lemma, 500-dim)")

except FileNotFoundError as e:
    print(f"ERROR: Files not found. {e}")
    exit()

# ===================================================================
# STEP 3: Run Cross-Validation
# ===================================================================
print(f"\nStep 3: Running {CV_FOLDS}-Fold Cross-Validation...")
model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)

print("  Testing Model 1 (V3, 100-dim)...")
scores_v3_100 = cross_val_score(model, X_buss_v3, y, cv=CV_FOLDS, n_jobs=N_JOBS, scoring='accuracy')

print("  Testing Model 2 (V4, 100-dim)...")
scores_v4_100 = cross_val_score(model, X_buss_v4_100, y, cv=CV_FOLDS, n_jobs=N_JOBS, scoring='accuracy')

print("  Testing Model 3 (V4, 500-dim)...")
scores_v4_500 = cross_val_score(model, X_buss_v4_500, y, cv=CV_FOLDS, n_jobs=N_JOBS, scoring='accuracy')

print("...All models tested.")

# ===================================================================
# STEP 4: Print Results
# ===================================================================
print("\n--- Pillar 2 Results: Isolating Variables ---")
print("                          |   Accuracy (Mean)   |   Std Dev")
print("--------------------------|---------------------|------------")
m1_acc = np.mean(scores_v3_100)
m2_acc = np.mean(scores_v4_100)
m3_acc = np.mean(scores_v4_500)
print(f" Model 1 (V3, 100-dim)    |   {m1_acc * 100:<17.2f} |   {np.std(scores_v3_100) * 100:.2f}")
print(f" Model 2 (V4, 100-dim)    |   {m2_acc * 100:<17.2f} |   {np.std(scores_v4_100) * 100:.2f}")
print(f" Model 3 (V4, 500-dim)    |   {m3_acc * 100:<17.2f} |   {np.std(scores_v4_500) * 100:.2f}")

# ===================================================================
# STEP 5: Statistical Verdict
# ===================================================================
print("\n--- Statistical Verdict (T-Tests) ---")

# Test 1: Effect of Lemmatization (Model 1 vs Model 2)
t_stat_lemma, p_val_lemma = ttest_ind(scores_v3_100, scores_v4_100)
effect_lemma = (m2_acc - m1_acc) * 100
print(f"  Effect of Lemmatization (M1 vs M2): {effect_lemma:+.2f}%")
print(f"  p-value (is diff significant?): {p_val_lemma:.4f}")
if p_val_lemma < 0.05:
    print("  VERDICT: Lemmatization had a STATISTICALLY SIGNIFICANT impact.")
else:
    print("  VERDICT: The difference is NOT statistically significant (p > 0.05).")


# Test 2: Effect of Extra Dimensions (Model 2 vs Model 3)
t_stat_dims, p_val_dims = ttest_ind(scores_v4_100, scores_v4_500)
effect_dims = (m3_acc - m2_acc) * 100
print(f"\n  Effect of 400 Extra Dims (M2 vs M3): {effect_dims:+.2f}%")
print(f"  p-value (is diff significant?): {p_val_dims:.4f}")
if p_val_dims < 0.05:
    print("  VERDICT: The 400 extra components had a STATISTICALLY SIGNIFICANT impact.")
else:
    print("  VERDICT: The difference is NOT statistically significant (p > 0.05).")

print("\n--- Pillar 2 Complete (Definitive) ---")