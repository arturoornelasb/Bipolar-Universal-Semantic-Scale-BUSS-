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
import matplotlib.pyplot as plt
import numpy as np
import os

print("--- Starting Visualization Script (Pillar 3) ---")

# --- Training Data (Extracted from terminal logs) ---
epochs = [1, 2, 3]

# Experiment 1 (Lambda = 0.01)
lm_loss_0_01 = [6.3005, 4.7974, 4.6833]
buss_loss_0_01 = [0.1209, 0.1229, 0.1224]

# Experiment 2 (Lambda = 0.1)
lm_loss_0_1 = [6.2937, 4.7959, 4.6861]
buss_loss_0_1 = [0.1203, 0.1224, 0.1216]

# Experiment 3 (Lambda = 1.0)
lm_loss_1_0 = [6.2639, 4.7916, 4.6889]
buss_loss_1_0 = [0.1208, 0.1222, 0.1223]

# ===================================================================
# --- Create the Plot ---
# ===================================================================
print("Generating loss curve plots...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# --- Plot 1: LM Loss ---
# FIX: Use raw strings (r'...') to ignore \l
ax1.plot(epochs, lm_loss_0_01, marker='o', linestyle='--', label=r'$\lambda=0.01$')
ax1.plot(epochs, lm_loss_0_1, marker='s', linestyle='--', label=r'$\lambda=0.1$')
ax1.plot(epochs, lm_loss_1_0, marker='^', linestyle='--', label=r'$\lambda=1.0$')
ax1.set_title('Pillar 3: Language Model Loss (LM Loss)', fontsize=16)
ax1.set_ylabel('Loss (Average Error)', fontsize=12)
ax1.set_xticks(epochs)
ax1.set_xticklabels(['Epoch 1', 'Epoch 2', 'Epoch 3'])
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

# --- Plot 2: BUSS Loss ---
# FIX: Use raw strings (r'...') to ignore \l
ax2.plot(epochs, buss_loss_0_01, marker='o', linestyle='-', label=r'$\lambda=0.01$')
ax2.plot(epochs, buss_loss_0_1, marker='s', linestyle='-', label=r'$\lambda=0.1$')
ax2.plot(epochs, buss_loss_1_0, marker='^', linestyle='-', label=r'$\lambda=1.0$')
ax2.set_title('Pillar 3: Bipolar Loss (BUSS Loss)', fontsize=16)
ax2.set_xlabel('Training Epoch', fontsize=12)
ax2.set_ylabel('Loss (Average Error)', fontsize=12)
ax2.set_xticks(epochs)
ax2.set_xticklabels(['Epoch 1', 'Epoch 2', 'Epoch 3'])
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.set_ylim(0.115, 0.125)

plt.tight_layout() 

plot_filename = "pillar3_loss_curves.png"
plt.savefig(plot_filename)

print(f"Plot saved as: {plot_filename}")
print("\n--- Pillar 3 Visualization Complete ---")