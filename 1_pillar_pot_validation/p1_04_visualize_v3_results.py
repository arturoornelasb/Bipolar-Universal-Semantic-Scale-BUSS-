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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import time

print("--- Starting Pillar 1 (V3 Visualization) ---")

# --- V3 Configuration ---
# We are loading the V3 files (100 components, simple stopwords)
output_dir = "data_output" 

# ===================================================================
# STEP 1: Load V3 matrices from disk
# ===================================================================
print("Loading V3 matrices (E_A, E_B, V_buss, V_lsa)...")
try:
    E_A = load_npz(os.path.join(output_dir, "E_A.npz")) # 1-star
    E_B = load_npz(os.path.join(output_dir, "E_B.npz")) # 5-star
    V_buss = np.load(os.path.join(output_dir, "V_buss.npy")) # BUSS axes (100)
    V_lsa = np.load(os.path.join(output_dir, "V_lsa.npy"))  # LSA axes (100)
except FileNotFoundError:
    print(f"ERROR: Files not found in '{output_dir}'.")
    print("Please ensure you ran the original 'p1_01' and 'p1_02' scripts.")
    exit()

print("All V3 matrices loaded.")

# ===================================================================
# STEP 2: Re-calculate V3 projections and centroids
# ===================================================================
print("Calculating V3 projections and centroids...")
start_time = time.time()

# --- BUSS Space ---
v_A_buss = (E_A @ V_buss).mean(axis=0) # Centroid for 1-star
v_B_buss = (E_B @ V_buss).mean(axis=0) # Centroid for 5-star

# --- LSA Space (Control) ---
v_A_lsa = (E_A @ V_lsa).mean(axis=0) # Centroid for 1-star
v_B_lsa = (E_B @ V_lsa).mean(axis=0) # Centroid for 5-star

print(f"Centroids calculated in {time.time() - start_time:.2f}s")

# ===================================================================
# STEP 3: Reduce 100-dim vectors to 2-dim (X, Y) using PCA
# ===================================================================
print("Running PCA to reduce 100-dim centroids to 2-dim...")

# Stack all 4 centroids into a single array (4, 100)
all_centroids = np.vstack([
    v_A_buss,
    v_B_buss,
    v_A_lsa,
    v_B_lsa
])

# Initialize PCA to find the 2 best components to plot
pca = PCA(n_components=2, random_state=42)
centroids_2d = pca.fit_transform(all_centroids)

# Separate the 2D points
p_A_buss = centroids_2d[0]
p_B_buss = centroids_2d[1]
p_A_lsa = centroids_2d[2]
p_B_lsa = centroids_2d[3]

# ===================================================================
# STEP 4: Create and Save the Plot
# ===================================================================
print("Creating plot...")

plt.figure(figsize=(10, 7))
ax = plt.subplot(111)

# Plot BUSS points (Red)
ax.scatter(p_A_buss[0], p_A_buss[1], c='red', s=100, label="BUSS (1-Star Centroid)")
ax.scatter(p_B_buss[0], p_B_buss[1], c='red', s=100, marker='x', label="BUSS (5-Star Centroid)")
# Draw a line between them
ax.plot([p_A_buss[0], p_B_buss[0]], [p_A_buss[1], p_B_buss[1]], 'r--', alpha=0.7)

# Plot LSA points (Blue)
ax.scatter(p_A_lsa[0], p_A_lsa[1], c='blue', s=100, label="LSA (1-Star Centroid)")
ax.scatter(p_B_lsa[0], p_B_lsa[1], c='blue', s=100, marker='x', label="LSA (5-Star Centroid)")
# Draw a line between them
ax.plot([p_A_lsa[0], p_B_lsa[0]], [p_A_lsa[1], p_B_lsa[1]], 'b--', alpha=0.7)

# --- Calculate distances for the title ---
dist_buss = np.linalg.norm(p_A_buss - p_B_buss)
dist_lsa = np.linalg.norm(p_A_lsa - p_B_lsa)

ax.set_title(
    f"Pillar 1: Centroid Separation (V3 Results, 100 Components)\n"
    f"BUSS 2D-Distance: {dist_buss:.4f}  |  LSA 2D-Distance: {dist_lsa:.4f}",
    fontsize=14
)
ax.set_xlabel("PCA Component 1", fontsize=12)
ax.set_ylabel("PCA Component 2", fontsize=12)
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

# Save the plot
plot_filename = "pillar1_v3_centroid_separation_plot.png"
plt.savefig(plot_filename)

print(f"\n--- Visualization Complete! ---")
print(f"Plot saved as: {plot_filename}")