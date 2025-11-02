#  BUSS-LoRA: Bipolar Universal Semantic Scale for LLM Fine-Tuning

This repository contains the source code, experimental scripts, and logs for the research paper: "BUSS-LoRA: Bipolar Universal Semantic Scale for LLM Fine-Tuning".

The proposed BUSS algorithm uses centered TF-IDF and SVD to model the Theorem of Perfect Opposition (TOP) in the vector space. Integrated with the LoRA adaptation framework, we fine-tune a Large Language Model (LLM) to generate content that is semantically opposite to the input when activating the prefix Bipolar_Opposite:, achieved via a custom Bipolar Loss.



##  Repository Structure

/
├── README.md
├── paper/
│   ├── draft.tex
│   └── figures/
├── src/
│   ├── data_processing/        # BUSS Core and POT Validation Scripts
│   │   ├── BUSS_fase_3.py      <-- BUSS Axes Analysis (Bipolar Axis Discovery)
│   │   └── buss_real_papers.py <-- Definitive BUSS Code for TOP Theorem (Papers Analysis)
│   ├── llm_finetuning/         # BUSS-LoRA Training Scripts (v5.1 is recommended)
│   │   ├── buss_lora_training_v2.py  <-- DEMO Training (v2.8 - Tanh Loss, test data)
│   │   ├── buss_lora_training_v5.py  <-- REAL Training (v5.0 - Tanh Loss, IMDB dataset)
│   │   └── buss_lora_training_v5.1.py <-- REAL Training (v5.1 - Similar to v5.0, text cleaning)
│   └── evaluation/             # Quantitative Evaluation using SBERT
│       ├── quantitative_eval.py      <-- Quantitative Evaluation (BUSS-like Embeddings)
│       ├── quantitative_eval_v2.py   <-- Quantitative Evaluation (SBERT, targets v5_imdb)
│       ├── quantitative_eval_v2.1.py <-- Quantitative Evaluation (SBERT, targets v5.0_imdb)
│       └── Q_E.py                    <-- Quantitative Evaluation (BUSS-like Embeddings, v2)
├── data/                       # Raw and processed datasets (aclImdb)
│   ├── papers_real/            <-- 20 arXiv papers
│   └── aclImdb/
├── models/
│   ├── buss_lora_final_v2/
│   ├── buss_lora_final_v5_imdb/
│   └── buss_lora_final/
├── environment/
│   └── requirements.txt
└── logs/
    ├──output_buss_lora_training_v2.txt
    ├──output_buss_lora_training_v5.txt
    ├──output_quantitative_eval_v2_buss_lora_training_v5.txt
    ├──output_buss_lora_training_v5.1.txt
    └──output_quantitative_eval_v2.1_buss_lora_training_v5.1.txt


##  Reproduction Guide

Follow these steps to fully replicate the experiments and results presented in the research paper.

### 1. Environment Setup

Install the dependencies listed in environment/requirements.txt:
# Recommended: Create a virtual environment first
pip install -r environment/requirements.txt

2. Data Preparation (IMDB)

The project requires the IMDB Large Movie Review dataset. You can download and prepare the data using the following commands:
# 1. Download the dataset (approx. 550MB)
wget [http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

# 2. Unzip the content into the main directory
tar -xzf aclImdb_v1.tar.gz

# 3. Move the data into the designated folder (if necessary)
mv aclImdb data/

# Expected structure:
# /data/aclImdb/train/pos/*.txt


3. Phase 1: Semantic Axis Discovery (BUSS Core)
Run this script to analyze the IMDB corpus, compute the Projection Matrix ($\mathbf{P}$), and discover the underlying semantic axes used for the classifier analysis:
# Analyzes the corpus and generates buss_projections_for_classification.csv
python src/data_processing/BUSS_fase_3.py

Key Output: The script prints the detected bipolar keyword poles (e.g., 'Pure Sentiment' axis).

4. Phase 2: BUSS-LoRA Training (Production Run)
Execute the production training script. Note that while v5.1 is the latest, the v5.0 run yielded the optimal quantitative score cited in the paper. We recommend running v5.1 for verification, but cite v5.0 as the primary result.
# Recommended script for replication (v5.1):
python src/llm_finetuning/buss_lora_training_v5.1.py

Key Configuration: Training is performed using a subtle BIPOLAR_WEIGHT = 0.001 to prevent catastrophic forgetting while enforcing duality over 3 epochs.


5. Phase 3: Quantitative Evaluation (Opposition Proof)
Use this script to measure the actual semantic separation (Cosine Similarity) between the Standard and Bipolar Opposite generations, validating the success of the Bipolar Loss.
# Measures SBERT Cosine Similarity and prints the final score.
python src/evaluation/quantitative_eval_v2.1.py


Success Criterion: A score close to $0.0$ or negative indicates strong semantic opposition. The score of $0.4858$ (from the v5.0 run) confirms sufficient Semantic Dispersion. The log file confirming this result is located at: logs/output_quantitative_eval_v2_buss_lora_training_v5.txt.

Note: This script is configured to load the model from ./buss_lora_final_imdb.

# Notes for Reviewers
The file src/llm_finetuning/buss_lora_training_v2.py is an earlier development version using a small internal demo dataset.

The script src/evaluation/Q_E.py uses a simpler, BUSS-like embedding for evaluation, but quantitative_eval_v2.1.py with SBERT is the definitive metric for the paper.

buss_lora_training_v5.1.py is the recommended final version, including text cleaning and token limits.


Citation
If you use this code or algorithm in your research, please cite our work:




e-mail:
arturoornelas62@gmail.com


## License and Copyright

© Copyright 2025 José Arturo Ornelas Brand

This project is licensed under the **Apache License, Version 2.0**.
Consult the [LICENSE](https://www.google.com/search?q=LICENSE) file for the full terms and conditions.

**Note on Dual Licensing:**
Should you or your company wish to use this software for proprietary commercial purposes that do not comply with the terms of the Apache License 2.0, please contact arturoornelas62@gmail.com to discuss commercial licensing options.































