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
import numpy.linalg as LA # <-- For calculating the norm
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_1samp 
import time
import os

print("--- Starting Pillar 1 (V4 - Steps 6, 7, & Metrics) ---")

# --- V4 Configuration ---
output_dir = "data_output_v4_lemma"

# ===================================================================
# STEP 1: Load all required V4 matrices from disk
# ===================================================================
print("Loading all lemmatized matrices (E_A, E_B, V_buss, V_lsa)...")
E_A = load_npz(os.path.join(output_dir, "E_A_lemma.npz")) # 1-star
E_B = load_npz(os.path.join(output_dir, "E_B_lemma.npz")) # 5-star
V_buss = np.load(os.path.join(output_dir, "V_buss_lemma.npy")) # BUSS axes
V_lsa = np.load(os.path.join(output_dir, "V_lsa_lemma.npy"))  # LSA axes

print("All matrices loaded.")
print(f"Shape V_buss (500 components): {V_buss.shape}")
print(f"Shape V_lsa (500 components): {V_lsa.shape}")

# ===================================================================
# STEP 6 & 7: Project partitions onto axes and get centroids
# ===================================================================
print("\nStep 6 & 7: Projecting partitions and calculating centroids...")
start_time = time.time()

# --- BUSS Space ---
# Project: (25000, 5000) @ (5000, 500) -> (25000, 500)
E_A_proj_buss = E_A @ V_buss
E_B_proj_buss = E_B @ V_buss
v_A_buss = E_A_proj_buss.mean(axis=0) # Centroid for 1-star
v_B_buss = E_B_proj_buss.mean(axis=0) # Centroid for 5-star

# --- LSA Space (Control) ---
E_A_proj_lsa = E_A @ V_lsa
E_B_proj_lsa = E_B @ V_lsa
v_A_lsa = E_A_proj_lsa.mean(axis=0) # Centroid for 1-star
v_B_lsa = E_B_proj_lsa.mean(axis=0) # Centroid for 5-star

print(f"Projections and centroids calculated in {time.time() - start_time:.2f}s")

# ===================================================================
# METRICS: Calculate Cosine Similarity & Norm of Sum
# ===================================================================
print("\n--- Final Metrics (Semantic Partitions) ---")

# --- Metric 1: Cosine Similarity ---
cos_sim_buss = cosine_similarity(v_A_buss.reshape(1, -1), v_B_buss.reshape(1, -1))[0][0]
cos_sim_lsa = cosine_similarity(v_A_lsa.reshape(1, -1), v_B_lsa.reshape(1, -1))[0][0]

# --- Metric 2: Norm of Sum (||v(A) + v(B)||) ---
norm_sum_buss = LA.norm(v_A_buss + v_B_buss)
norm_sum_lsa = LA.norm(v_A_lsa + v_B_lsa)

print("                      BUSS (V4)   |   LSA (V4)")
print(f"Cosine Similarity:    {cos_sim_buss:<10.6f}  |   {cos_sim_lsa:<10.6f}")
print(f"Norm of Sum (||A+B||):  {norm_sum_buss:<10.6f}  |   {norm_sum_lsa:<10.6f}")

if (cos_sim_buss < cos_sim_lsa) and (norm_sum_buss < norm_sum_lsa):
    print("SUCCESS: BUSS shows better separation on BOTH metrics.")
elif cos_sim_buss < cos_sim_lsa:
    print("SUCCESS: BUSS shows better Cosine separation.")
elif norm_sum_buss < norm_sum_lsa:
    print("SUCCESS: BUSS shows better Norm of Sum separation.")
else:
    print("FAILURE: LSA performed better on both metrics.")

# ===================================================================
# BONUS: Random Control & T-Test (V4)
# ===================================================================
print("\n--- Bonus: Random Control & T-Test (V4) ---")
print("Running 10 trials with random partitions...")

n_reviews = E_A.shape[0] # 25000
all_indices = np.arange(n_reviews)
buss_random_cos = []
lsa_random_cos = []
buss_random_norm = []
lsa_random_norm = []

for i in range(10):
    np.random.shuffle(all_indices)
    indices_C = all_indices[:n_reviews // 2]
    indices_D = all_indices[n_reviews // 2:]
    E_C, E_D = E_A[indices_C], E_A[indices_D] # Split 1-star reviews randomly
    
    # --- BUSS (Random) ---
    v_C_buss = (E_C @ V_buss).mean(axis=0)
    v_D_buss = (E_D @ V_buss).mean(axis=0)
    buss_random_cos.append(cosine_similarity(v_C_buss.reshape(1, -1), v_D_buss.reshape(1, -1))[0][0])
    buss_random_norm.append(LA.norm(v_C_buss + v_D_buss))
    
    # --- LSA (Random) ---
    v_C_lsa = (E_C @ V_lsa).mean(axis=0)
    v_D_lsa = (E_D @ V_lsa).mean(axis=0)
    lsa_random_cos.append(cosine_similarity(v_C_lsa.reshape(1, -1), v_D_lsa.reshape(1, -1))[0][0])
    lsa_random_norm.append(LA.norm(v_C_lsa + v_D_lsa))

# Calculate stats for random controls
print("\n--- Random Control Metrics (Mean ± Std) ---")
print("                      BUSS (V4)             |   LSA (V4)")
print(f"Cosine Similarity:    {np.mean(buss_random_cos):.6f} ± {np.std(buss_random_cos):.6f}   |   {np.mean(lsa_random_cos):.6f} ± {np.std(lsa_random_cos):.6f}")
print(f"Norm of Sum (||A+B||):  {np.mean(buss_random_norm):.6f} ± {np.std(buss_random_norm):.6f}   |   {np.mean(lsa_random_norm):.6f} ± {np.std(lsa_random_norm):.6f}")

# --- Statistical Significance (T-Test) ---
print("\n--- T-Test (Semantic vs. Random Control) ---")
# --- NEW: Using ttest_1samp ---
# Compares our sample of random scores to a single population mean (our semantic score)
t_cos_buss, p_cos_buss = ttest_1samp(buss_random_cos, cos_sim_buss)
t_cos_lsa, p_cos_lsa = ttest_1samp(lsa_random_cos, cos_sim_lsa)
t_norm_buss, p_norm_buss = ttest_1samp(buss_random_norm, norm_sum_buss)
t_norm_lsa, p_norm_lsa = ttest_1samp(lsa_random_norm, norm_sum_lsa)

print("                      p-value (BUSS) | p-value (LSA)")
print(f"Cosine Similarity:    {p_cos_buss:<10.2e}     |   {p_cos_lsa:<10.2e}")
print(f"Norm of Sum (||A+B||):  {p_norm_buss:<10.2e}     |   {p_norm_lsa:<10.2e}")
print("(p-value < 0.05 means the semantic score is statistically significant)")

print("\n--- Pillar 1 V4 Complete ---")