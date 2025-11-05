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
from datasets import load_dataset
import pickle
import time
import os

print("--- Starting Pillar 3 (Prep): Building the Final Classifier ---")

# --- Configuration ---
V4_DIR = "data_output_v4_lemma"
CLASSIFIER_OUTPUT_FILE = os.path.join(V4_DIR, "buss_v4_classifier.pkl")

# ===================================================================
# STEP 1: Load V4 Data (X) and Labels (y)
# ===================================================================
print("Step 1: Loading V4 data and labels...")
start_time = time.time()
try:
    # Load V4 Data
    X_tfidf_v4 = load_npz(os.path.join(V4_DIR, "E_lemma.npz"))
    V_buss_v4 = np.load(os.path.join(V4_DIR, "V_buss_lemma.npy")) # (5000, 500)
    
    # Create V4 projection features
    X_features = X_tfidf_v4 @ V_buss_v4
    print(f"  V4 features (X) loaded. Shape: {X_features.shape}")

    # Load Labels
    imdb_train = load_dataset("imdb", split="train")
    imdb_test = load_dataset("imdb", split="test")
    y_labels = np.array(list(imdb_train['label']) + list(imdb_test['label']))
    print(f"  Labels (y) loaded. Shape: {y_labels.shape}")

except FileNotFoundError as e:
    print(f"ERROR: Files not found. {e}")
    exit()

print(f"Data loaded in {time.time() - start_time:.2f}s")

# ===================================================================
# STEP 2: Train the Final Classifier
# ===================================================================
print("\nStep 2: Training final Logistic Regression on all 50k samples...")
start_time = time.time()

# Train on all 50,000 samples to get the definitive classifier
# We increase max_iter for convergence on the full dataset
model = LogisticRegression(max_iter=1500, random_state=42, n_jobs=-1)
model.fit(X_features, y_labels)

print(f"Model trained in {time.time() - start_time:.2f}s")

# ===================================================================
# STEP 3: Save the Classifier
# ===================================================================
print(f"\nStep 3: Saving classifier to '{CLASSIFIER_OUTPUT_FILE}'...")

with open(CLASSIFIER_OUTPUT_FILE, "wb") as f:
    pickle.dump(model, f)

print("Classifier saved successfully.")
print("\n--- Pillar 3 Preparation Complete ---")
print("We are now ready for the final evaluation script.")