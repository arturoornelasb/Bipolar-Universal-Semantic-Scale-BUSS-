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
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import os
import time
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Download NLTK data (only need to run once) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer data...")
    nltk.download('punkt')
    
# --- NEW: Add the missing dependency ---
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK 'punkt_tab' dependency...")
    nltk.download('punkt_tab')
# --- End of new code ---

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK 'wordnet' lemmatizer data...")
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK 'stopwords' data...")
    nltk.download('stopwords')

# --- V4: Define a custom lemmatizer tokenizer ---
lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def lemma_tokenizer(text):
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            lemmatized_tokens.append(lemmatizer.lemmatize(token))
    return lemmatized_tokens

print("--- Starting Pillar 1 (V4 - Lemmatized) ---")

output_dir = "data_output_v4_lemma"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# ===================================================================
# STEP 1: Load IMDB (50k)
# ===================================================================
print("Step 1: Loading IMDB (50k) dataset...")
start_time = time.time()

imdb_train = load_dataset("imdb", split="train")
imdb_test = load_dataset("imdb", split="test")

all_texts = list(imdb_train['text']) + list(imdb_test['text'])
all_labels = np.array(list(imdb_train['label']) + list(imdb_test['label']))

print(f"Load complete! {len(all_texts)} documents loaded in {time.time() - start_time:.2f}s")

# ===================================================================
# STEP 2: Create Matrix E (TF-IDF 5k, Lemmatized)
# ===================================================================
print("\nStep 2: Creating Matrix E (Lemmatized TF-IDF)...")
print("(This will take significantly longer due to custom tokenizer)")
start_time = time.time()

tfidf_vectorizer_v4 = TfidfVectorizer(
    tokenizer=lemma_tokenizer,
    max_features=5000
)

E = tfidf_vectorizer_v4.fit_transform(all_texts)

print(f"Matrix E (lemmatized) created in {time.time() - start_time:.2f}s!")
print(f"Shape of E (documents, features): {E.shape}")

# ===================================================================
# PREPARATION
# ===================================================================
print("\nPreparing partitions...")

mask_A = (all_labels == 0) # 1-star
mask_B = (all_labels == 1) # 5-star

E_A = E[mask_A]
E_B = E[mask_B]

print(f"Shape of E_A (1-star/neg): {E_A.shape}")
print(f"Shape of E_B (5-star/pos): {E_B.shape}")

# ===================================================================
# STEP 2.5: Save matrices to new V4 directory
# ===================================================================
print("\nStep 2.5: Saving matrices to disk...")

save_npz(os.path.join(output_dir, "E_lemma.npz"), E)
save_npz(os.path.join(output_dir, "E_A_lemma.npz"), E_A)
save_npz(os.path.join(output_dir, "E_B_lemma.npz"), E_B)

import pickle
with open(os.path.join(output_dir, "tfidf_vectorizer_v4.pkl"), "wb") as f:
    pickle.dump(tfidf_vectorizer_v4, f)

print(f"Successfully saved matrices to '{output_dir}'")
print("\n--- V4 Steps 1 and 2 completed! ---")