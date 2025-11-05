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
import pickle
import os
import matplotlib.pyplot as plt

# --- Add NLTK imports and function definition ---
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
import nltk.corpus 
stop_words = set(nltk.corpus.stopwords.words('english'))

def lemma_tokenizer(text):
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            lemmatized_tokens.append(lemmatizer.lemmatize(token))
    return lemmatized_tokens
# --- END OF FIX ---


print("--- Starting Pillar 2 (Nomenclature): Validating Sentiment Axis ---")

# --- Configuration ---
V4_DIR = "data_output_v4_lemma"
N_WORDS_TO_SHOW = 15 

# ===================================================================
# STEP 1: Load Vectorizer and Sentiment Axis
# ===================================================================
print("Step 1: Loading vectorizer and sentiment axis...")
try:
    with open(os.path.join(V4_DIR, "tfidf_vectorizer_v4.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    
    axis_vector = np.load(os.path.join(V4_DIR, "sentiment_axis_vector.npy"))
    axis_index = np.load(os.path.join(V4_DIR, "sentiment_axis_index.npy"))

except FileNotFoundError as e:
    print(f"ERROR: Files not found in '{V4_DIR}'. {e}")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

print(f"Successfully loaded vectorizer (vocab size: {len(vectorizer.vocabulary_)})")
print(f"Successfully loaded sentiment axis #{axis_index} (shape: {axis_vector.shape})")

# ===================================================================
# STEP 2: Map Indices to Feature Names (Words)
# ===================================================================
feature_names = {i: word for word, i in vectorizer.vocabulary_.items()}

# ===================================================================
# STEP 3: Find Top and Bottom Words (and weights)
# ===================================================================
print("\nStep 3: Finding top and bottom words for the axis...")

sorted_indices = np.argsort(axis_vector)

# --- Pole A (Low values) ---
negative_pole_indices = sorted_indices[:N_WORDS_TO_SHOW]
negative_pole_words = [feature_names[i] for i in negative_pole_indices]
negative_pole_weights = [axis_vector[i] for i in negative_pole_indices]

# --- Pole B (High values) ---
positive_pole_indices = sorted_indices[-N_WORDS_TO_SHOW:][::-1] 
positive_pole_words = [feature_names[i] for i in positive_pole_indices]
positive_pole_weights = [axis_vector[i] for i in positive_pole_indices]

# ===================================================================
# STEP 4: Print the Axis Nomenclature (with Flipping Logic)
# ===================================================================
print(f"\n--- Nomenclature for BUSS Axis #{axis_index} (The 'Sentiment Axis') ---")

# --- UPDATED: Expanded keyword list for flipping ---
negative_keywords = {'bad', 'worst', 'terrible', 'awful', 'stupid', 'waste'}
# Check if any of these keywords are in the pole with high (positive) weights
is_inverted = any(word in positive_pole_words for word in negative_keywords)

if is_inverted:
    print("--- Axis validation: INVERTED (Positive weights = Negative sentiment) ---")
    pole_A_label = "--- Top 15 Positive Sentiment Words (Low Weights) ---"
    pole_A_words = negative_pole_words
    pole_A_weights = negative_pole_weights
    
    pole_B_label = "--- Top 15 Negative Sentiment Words (High Weights) ---"
    pole_B_words = positive_pole_words
    pole_B_weights = positive_pole_weights
else:
    print("--- Axis validation: ALIGNED (Positive weights = Positive sentiment) ---")
    pole_A_label = "--- Top 15 Positive Sentiment Words (High Weights) ---"
    pole_A_words = positive_pole_words
    pole_A_weights = positive_pole_weights

    pole_B_label = "--- Top 15 Negative Sentiment Words (Low Weights) ---"
    pole_B_words = negative_pole_words
    pole_B_weights = negative_pole_weights

# --- Print the re-labeled poles with weights ---
print(f"\n   {pole_A_label}")
for i in range(N_WORDS_TO_SHOW):
    print(f"   {i+1:>2}. {pole_A_words[i]:<15} ({pole_A_weights[i]:.4f})")

print(f"\n   {pole_B_label}")
for i in range(N_WORDS_TO_SHOW):
    print(f"   {i+1:>2}. {pole_B_words[i]:<15} ({pole_B_weights[i]:.4f})")
    
# ===================================================================
# STEP 5: Create and Save Plot 
# ===================================================================
print("\nStep 5: Generating visualization...")

# Combine the words and weights for plotting
# We reverse Pole A so the plot goes from "Most Positive" to "Most Negative"
plot_words = pole_A_words[::-1] + pole_B_words
plot_weights = pole_A_weights[::-1] + pole_B_weights
# Create a color list: green for positive (low weights), red for negative (high weights)
colors = ['green'] * N_WORDS_TO_SHOW + ['red'] * N_WORDS_TO_SHOW

plt.figure(figsize=(12, 10))
plt.barh(plot_words, plot_weights, color=colors)
plt.xlabel("Weight on Axis #4 (Sentiment)", fontsize=12)
plt.ylabel("Top Words", fontsize=12)
plt.title(f"Pillar 2: BUSS Axis #{axis_index} Nomenclature", fontsize=16)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout() # Adjust plot to prevent labels from being cut off

plot_filename = "pillar2_axis_nomenclature.png"
plt.savefig(plot_filename)

print(f"Plot saved as: {plot_filename}")
print("\n--- Pillar 2 Complete (Definitive) ---")