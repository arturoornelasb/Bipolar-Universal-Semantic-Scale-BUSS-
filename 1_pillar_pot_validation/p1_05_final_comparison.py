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
import numpy.linalg as LA
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_1samp
import time
import os

def run_analysis(experiment_id, data_dir, e_file, ea_file, eb_file, vb_file, vl_file):
    """
    Runs the full Pillar 1 analysis for a given set of experiment files.
    """
    print(f"\n--- Running Analysis for: {experiment_id} ---")
    
    # ===================================================================
    # 1. Load all required matrices from disk
    # ===================================================================
    print(f"Loading files from: {data_dir}")
    try:
        E = load_npz(os.path.join(data_dir, e_file))     # Full corpus E
        E_A = load_npz(os.path.join(data_dir, ea_file)) # 1-star
        E_B = load_npz(os.path.join(data_dir, eb_file)) # 5-star
        V_buss = np.load(os.path.join(data_dir, vb_file)) # BUSS axes
        V_lsa = np.load(os.path.join(data_dir, vl_file))  # LSA axes
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure all previous scripts ran successfully.")
        return None

    print(f"Axes shape (BUSS): {V_buss.shape}")
    print(f"Axes shape (LSA):  {V_lsa.shape}")

    # ===================================================================
    # 2. Calculate Semantic Metrics (Opposed Partitions)
    # ===================================================================
    print("Calculating semantic (opposed) centroids...")
    # --- BUSS Space ---
    v_A_buss = (E_A @ V_buss).mean(axis=0) # 1-star
    v_B_buss = (E_B @ V_buss).mean(axis=0) # 5-star
    # --- LSA Space ---
    v_A_lsa = (E_A @ V_lsa).mean(axis=0) # 1-star
    v_B_lsa = (E_B @ V_lsa).mean(axis=0) # 5-star

    # --- Metric 1: Cosine Similarity ---
    sem_cos_buss = cosine_similarity(v_A_buss.reshape(1, -1), v_B_buss.reshape(1, -1))[0][0]
    sem_cos_lsa = cosine_similarity(v_A_lsa.reshape(1, -1), v_B_lsa.reshape(1, -1))[0][0]
    # --- Metric 2: Norm of Sum ---
    sem_norm_buss = LA.norm(v_A_buss + v_B_buss)
    sem_norm_lsa = LA.norm(v_A_lsa + v_B_lsa)

    # ===================================================================
    # 3. Calculate Random Control Metrics (Corrected Method)
    # ===================================================================
    print("Calculating random control (10 trials)...")
    # --- NEW: Using the full corpus E, as per reviewer suggestion ---
    n_docs = E.shape[0] # 50000
    n_half = n_docs // 2  # 25000
    
    buss_random_cos, lsa_random_cos = [], []
    buss_random_norm, lsa_random_norm = [], []

    for i in range(10):
        # Create random partitions C and D from the *full* corpus
        indices = np.random.permutation(n_docs)
        indices_C, indices_D = indices[:n_half], indices[n_half:]
        E_C, E_D = E[indices_C], E[indices_D]
        
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

    # ===================================================================
    # 4. Calculate T-Tests (Corrected Method)
    # ===================================================================
    t_cos_buss, p_cos_buss = ttest_1samp(buss_random_cos, sem_cos_buss)
    t_cos_lsa, p_cos_lsa = ttest_1samp(lsa_random_cos, sem_cos_lsa)
    t_norm_buss, p_norm_buss = ttest_1samp(buss_random_norm, sem_norm_buss)
    t_norm_lsa, p_norm_lsa = ttest_1samp(lsa_random_norm, sem_norm_lsa)
    
    # ===================================================================
    # 5. Print Results
    # ===================================================================
    print("\n--- FINAL RESULTS TABLE ---")
    print(f"--- {experiment_id} ---")
    print("                      |   BUSS (Semantic)  |    LSA (Semantic)  | BUSS (Random) Mean | LSA (Random) Mean")
    print(f"Cosine Similarity     |   {sem_cos_buss:<16.6f} |   {sem_cos_lsa:<16.6f} |   {np.mean(buss_random_cos):<16.6f} |   {np.mean(lsa_random_cos):<16.6f}")
    print(f"Norm of Sum (||A+B||) |   {sem_norm_buss:<16.6f} |   {sem_norm_lsa:<16.6f} |   {np.mean(buss_random_norm):<16.6f} |   {np.mean(lsa_random_norm):<16.6f}")
    print("\n--- p-values (Semantic vs. Random) ---")
    print(f"Cosine (p-value)      |   {p_cos_buss:<16.2e} |   {p_cos_lsa:<16.2e}")
    print(f"Norm (p-value)        |   {p_norm_buss:<16.2e} |   {p_norm_lsa:<16.2e}")
    print("---------------------------------------\n")
    
    # Return a dictionary of the key results for final comparison
    return {
        "id": experiment_id,
        "cos_buss": sem_cos_buss, "cos_lsa": sem_cos_lsa,
        "norm_buss": sem_norm_buss, "norm_lsa": sem_norm_lsa
    }

# ===================================================================
# --- MAIN EXECUTION ---
# ===================================================================

# Run V3 (100 components, simple stopwords)
v3_results = run_analysis(
    experiment_id="V3 (100 Comp, Stopwords)",
    data_dir="data_output",
    e_file="E.npz", ea_file="E_A.npz", eb_file="E_B.npz",
    vb_file="V_buss.npy", vl_file="V_lsa.npy"
)

# Run V4 (500 components, lemmatized)
v4_results = run_analysis(
    experiment_id="V4 (500 Comp, Lemmatized)",
    data_dir="data_output_v4_lemma",
    e_file="E_lemma.npz", ea_file="E_A_lemma.npz", eb_file="E_B_lemma.npz",
    vb_file="V_buss_lemma.npy", vl_file="V_lsa_lemma.npy"
)

# ===================================================================
# --- FINAL VERDICT ---
# ===================================================================
print("\n--- FINAL VERDICT: V3 vs V4 ---")
if v3_results and v4_results:
    # Calculate the separation "gap" (advantage of BUSS over LSA)
    v3_cos_gap = v3_results["cos_lsa"] - v3_results["cos_buss"]
    v4_cos_gap = v4_results["cos_lsa"] - v4_results["cos_buss"]
    
    v3_norm_gap = v3_results["norm_lsa"] - v3_results["norm_buss"]
    v4_norm_gap = v4_results["norm_lsa"] - v4_results["norm_buss"]

    print("Metric: Advantage Gap (LSA - BUSS), higher is better")
    print("                      |   V3 (100 Comp)    |   V4 (500 Comp)")
    print(f"Cosine Gap            |   {v3_cos_gap:<16.6f} |   {v4_cos_gap:<16.6f}")
    print(f"Norm of Sum Gap       |   {v3_norm_gap:<16.6f} |   {v4_norm_gap:<16.6f}")

    if v3_cos_gap > v4_cos_gap and v3_norm_gap > v4_norm_gap:
        print("\nVERDICT: V3 (100 Comp) is the definitive winner on both metrics.")
    elif v3_cos_gap > v4_cos_gap:
        print("\nVERDICT: V3 (100 Comp) is the winner on Cosine separation.")
    elif v3_norm_gap > v4_norm_gap:
        print("\nVERDICT: V3 (100 Comp) is the winner on Norm of Sum separation.")
    else:
        print("\nVERDICT: V4 (500 Comp) is the definitive winner.")
else:
    print("One of the analyses failed. Cannot compute final verdict.")

print("\n--- Pillar 1 Complete (Definitive) ---")