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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import time
import os

print("--- Starting Pillar 1 (V4 - Steps 3, 4, & 5) ---")

# --- V4 Configuration ---
output_dir = "data_output_v4_lemma"
N_COMPONENTS = 500  

# ===================================================================
# Load E (lemmatized) from disk
# ===================================================================
print("Loading sparse matrix E (lemmatized) from disk...")
E = load_npz(os.path.join(output_dir, "E_lemma.npz"))
print(f"Matrix E loaded. Shape: {E.shape}")

# ===================================================================
# STEP 3: Create Matrix Ec (Centered E)
# ===================================================================
print("\nStep 3: Centering Matrix E to create Ec...")
print("(This step will use more RAM as it creates a dense matrix)")
start_time = time.time()

# 1. Convert the sparse matrix E to a dense numpy array
E_dense = E.toarray()

# 2. Now, center the DENSE matrix.
scaler = StandardScaler(with_std=False) # with_std=False only subtracts mean
E_c = scaler.fit_transform(E_dense)

del E_dense 

print(f"Matrix Ec (dense) created in {time.time() - start_time:.2f}s")
print(f"Shape of Ec: {E_c.shape}")

# ===================================================================
# STEP 4: Get BUSS Axes (V_buss) from SVD on Ec
# ===================================================================
print(f"\nStep 4: Running TruncatedSVD on Ec (n={N_COMPONENTS})...")
print("(This is the BUSS calculation and will take several minutes...)")
start_time = time.time()

svd_buss = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
svd_buss.fit(E_c) 

V_buss_T = svd_buss.components_
V_buss = V_buss_T.T 

print(f"BUSS axes (V_buss) calculated in {time.time() - start_time:.2f}s")
print(f"Shape of V_buss: {V_buss.shape}")

del E_c

# ===================================================================
# STEP 5: Get LSA Axes (V_lsa) from SVD on E
# ===================================================================
print(f"\nStep 5: Running TruncatedSVD on E (n={N_COMPONENTS})...")
print("(This is the LSA calculation...)")
start_time = time.time()

svd_lsa = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
svd_lsa.fit(E)

V_lsa_T = svd_lsa.components_
V_lsa = V_lsa_T.T 

print(f"LSA axes (V_lsa) calculated in {time.time() - start_time:.2f}s")
print(f"Shape of V_lsa: {V_lsa.shape}")

# ===================================================================
# STEP 5.5: Save axes to disk
# ===================================================================
print("\nStep 5.5: Saving axes to disk...")

np.save(os.path.join(output_dir, "V_buss_lemma.npy"), V_buss)
np.save(os.path.join(output_dir, "V_lsa_lemma.npy"), V_lsa)

print(f"Successfully saved V_buss_lemma.npy and V_lsa_lemma.npy to '{output_dir}'")

print("\n--- V4 Steps 3, 4, and 5 completed! ---")