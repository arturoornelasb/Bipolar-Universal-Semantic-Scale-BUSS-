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
import time
import os

print("--- Starting Pillar 2 (Final Step): Finding Sentiment Axis ---")

# --- Configuration ---
V4_DIR = "data_output_v4_lemma"

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
    X_buss_v4_500 = X_tfidf_v4 @ V_buss_v4
    print(f"  V4 features (X) loaded. Shape: {X_buss_v4_500.shape}")

    # Load Labels
    imdb_train = load_dataset("imdb", split="train")
    imdb_test = load_dataset("imdb", split="test")
    y = np.array(list(imdb_train['label']) + list(imdb_test['label']))
    print(f"  Labels (y) loaded. Shape: {y.shape}")

except FileNotFoundError as e:
    print(f"ERROR: Files not found. {e}")
    exit()

print(f"Data loaded in {time.time() - start_time:.2f}s")

# ===================================================================
# STEP 2: Train Classifier to find coefficients
# ===================================================================
print("\nStep 2: Training final Logistic Regression on all 50k samples...")
start_time = time.time()

# Train on all 50,000 samples to get the definitive coefficients
model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
model.fit(X_buss_v4_500, y)

print(f"Model trained in {time.time() - start_time:.2f}s")

# ===================================================================
# STEP 3: Find the Axis with the Max Coefficient
# ===================================================================
print("\nStep 3: Finding the 'Pure Sentiment Axis'...")

# model.coef_ shape is (1, 500). We get the first row.
coefficients = model.coef_[0]

# Find the index of the coefficient with the largest *absolute* value
sentiment_axis_index = np.argmax(np.abs(coefficients))
sentiment_axis_weight = coefficients[sentiment_axis_index]

print(f"  --- Found! ---")
print(f"  Axis Index: {sentiment_axis_index}")
print(f"  Axis Weight (Coefficient): {sentiment_axis_weight:.4f}")

# The axis itself is the column from our V_buss matrix
sentiment_axis_vector = V_buss_v4[:, sentiment_axis_index]

# ===================================================================
# STEP 4: Save the axis for Pillar 3
# ===================================================================
print("\nStep 4: Saving the axis vector for Pillar 3...")

# We'll save both the index and the vector itself
np.save(os.path.join(V4_DIR, "sentiment_axis_index.npy"), sentiment_axis_index)
np.save(os.path.join(V4_DIR, "sentiment_axis_vector.npy"), sentiment_axis_vector)

print(f"Successfully saved axis {sentiment_axis_index} to '{V4_DIR}'")
print("\n--- Pillar 2 Complete (Definitive) ---")
print("We are now ready to begin Pillar 3.")