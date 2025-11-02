#  BUSS-LoRA: Bipolar Universal Semantic Scale for LLM Fine-Tuning

This repository contains the source code, experimental scripts, and logs for the research paper: "BUSS-LoRA: Bipolar Universal Semantic Scale for LLM Fine-Tuning".

The proposed BUSS algorithm uses centered TF-IDF and SVD to model the Theorem of Perfect Opposition (TOP) in the vector space. Integrated with the LoRA adaptation framework, we fine-tune a Large Language Model (LLM) to generate content that is semantically opposite to the input when activating the prefix Bipolar_Opposite:, achieved via a custom Bipolar Loss.



##  Repository Structure
```text
/
├── README.md
├── paper/
│   ├── draft.tex
│   └── figures/
├── src/
│   ├── data_processing/        # BUSS Core and POT Validation Scripts
│   │   ├── BUSS_fase_3.py                   <-- BUSS Axes Analysis (Bipolar Axis Discovery)
│   │   ├── buss_real_papers.py              <-- Definitive BUSS Code for TOP Theorem (Papers Analysis)
│   │   ├─  buss_codependence_classifier.py  <-- accuracy_score and classification_report (output _generate_loss_plot.txt)
│   │   └─  buss_final_semantic_map.py  <-- generate buss_projections_for_classification.csv
│   ├── llm_finetuning/         # BUSS-LoRA Training Scripts (v5.1 is recommended)
│   │   ├── buss_lora_training_v2.py  <-- DEMO Training (v2.8 - Tanh Loss, test data)
│   │   ├── buss_lora_training_v5.py  <-- REAL Training (v5.0 - Tanh Loss, IMDB dataset)
│   │   └── buss_lora_training_v5.1.py <-- REAL Training (v5.1 - Similar to v5.0, text cleaning)
│   └── evaluation/             # Quantitative Evaluation using SBERT
│       ├── quantitative_eval.py      <-- Quantitative Evaluation (BUSS-like Embeddings)
│       ├── quantitative_eval_v2.py   <-- Quantitative Evaluation (SBERT, targets v5_imdb)
│       ├── quantitative_eval_v2.1.py <-- Quantitative Evaluation (SBERT, targets v5.0_imdb)
│       ├── Q_E.py                    <-- Quantitative Evaluation (BUSS-like Embeddings, v2)
│       └── baseline_eval.py          <-- Quantitative Evaluation of the BASELINE MODEL
├── data/                       # Raw and processed datasets (aclImdb)
│   ├── /buss_projections_for_classification.csv   <-- output buss_final_semantic_map.py
│   ├── papers_real/                               <-- 20 arXiv papers
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
    ├──output_quantitative_eval_v2.1_buss_lora_training_v5.1.txt
    ├──output _generate_loss_plot.txt
    └──output_baseline_eval.txt
```

##  Reproduction Guide

Follow these steps to fully replicate the experiments and results presented in the research paper.

### 1. Environment Setup

Install the dependencies listed in environment/requirements.txt:
# Recommended: Create a virtual environment first
pip install -r environment/requirements.txt

2. Data Preparation (IMDB) and 20 arXiv papers

eference Format (MLA/APA):
Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis.

Full BibTeX Format:

@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {[http://www.aclweb.org/anthology/P11-1015](http://www.aclweb.org/anthology/P11-1015)}
}

The project requires the IMDB Large Movie Review dataset. You can download and prepare the data using the following commands:
# 1. Download the dataset (approx. 550MB)
wget [http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

# 2. Unzip the content into the main directory
tar -xzf aclImdb_v1.tar.gz

# 3. Move the data into the designated folder (if necessary)
mv aclImdb data/

# Expected structure:
# /data/aclImdb/train/pos/*.txt


# N.º	ID arXiv	Títle, Author link
1	2510.25687v1	Model Inversion Attacks Meet Cryptographic Fuzzy Extractors. Mallika Prabhakar, Louise Xu, Prateek Saxena. https://arxiv.org/abs/2510.25687
2	2510.25657v1	Subgraph Federated Learning via Spectral Methods. Javad Aliakbari, Johan Östman, Ashkan Panahi, Alexandre Graell i Amat. https://arxiv.org/abs/2510.25657
3	2510.25557v1	Hybrid Quantum-Classical Recurrent Neural Networks. Wenduan Xu. https://arxiv.org/abs/2510.25557
4	2510.25361v1	Parameter Averaging in Link Prediction. Rupesh Sapkota, Caglar Demir, Arnab Sharma, Axel-Cyrille Ngonga Ngomo. https://arxiv.org/abs/2510.25361
5	2510.25347v1	3D CT-Based Coronary Calcium Assessment: A Feature-Driven Machine Learning Framework. Ayman Abaid, Gianpiero Guidone, Sara Alsubai, Foziyah Alquahtani, Talha Iqbal, Ruth Sharif, Hesham Elzomor, Emiliano Bianchini, Naeif Almagal, Michael G. Madden, Faisal Sharif, Ihsan Ullah. https://arxiv.org/abs/2510.25347
6	2510.25262v1	IBNorm: Information-Bottleneck Inspired Normalization for Representation Learning. Xiandong Zou, Pan Zhou. https://arxiv.org/abs/2510.25262
7	2510.25182v1	Retaining Mixture Representations for Domain Generalized Anomalous Sound Detection. Phurich Saengthong, Tomoya Nishida, Kota Dohi, Natsuo Yamashita, Yohei Kawaguchi. https://arxiv.org/abs/2510.25182
8	2510.25105v1	Learning-Based vs Human-Derived Congestion Control: An In-Depth Experimental Study. Mihai Mazilu, Luca Giacomoni, George Parisis. https://arxiv.org/abs/2510.25105
9	2510.24951v1	Resource-Efficient and Robust Inference of Deep and Bayesian Neural Networks on Embedded and Analog Computing Platforms. Bernhard Klein. https://arxiv.org/abs/2510.24951
10	2510.24829v2	Send Less, Save More: Energy-Efficiency Benchmark of Embedded CNN Inference vs. Data Transmission in IoT. Benjamin Karic, Nina Herrmann, Jan Stenkamp, Paula Scharf, Fabian Gieseke, Angela Schwering. https://arxiv.org/abs/2510.24829
11	2510.24802v1	From Narrative to Action: A Hierarchical LLM-Agent Framework for Human Mobility Generation. Qiumeng Li, Chunhou Ji, Xinyue Liu. https://arxiv.org/abs/2510.24802
12	2510.24709v1	Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers? Yihao Li, Saeed Salehi, Lyle Ungar, Konrad P. https://arxiv.org/abs/2510.24709
13	2510.24523v1	Unsupervised Machine-Learning Pipeline for Data-Driven Defect Detection and Characterisation: Application to Displacement Cascades. Samuel Del Fré, Andrée de Backer, Christophe Domain, Ludovic Thuinet, Charlotte S. Becquart. https://arxiv.org/abs/2510.24523
14	2510.24262v1	UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation. Jiyu Guo, Shuo Yang, Yiming Huang, Yancheng Long, Xiaobo Xia, Xiu Su, Bo Zhao, Zeke Xie, Liqiang Nie. https://arxiv.org/abs/2510.24262
15	2510.24233v1	PRIVET: Privacy Metric Based on Extreme Value Theory. Antoine Szatkownik, Aurélien Decelle, Beatriz. https://arxiv.org/abs/2510.24233
16	2510.24058v1	PULSE: Privileged Knowledge Transfer from Electrodermal Activity to Low-Cost Sensors for Stress Monitoring. Zihan Zhao, Masood Mortazavi, Ning Yan. https://arxiv.org/abs/2510.24058
17	2510.24027v1	Spatio-temporal Multivariate Time Series Forecast with Chosen Variables. Zibo Liu, Zhe Jiang, Zelin Xu, Tingsong Xiao, Yupu Zhang, Zhengkun Xiao, Haibo Wang, Shigang Chen. https://arxiv.org/abs/2510.24027
18	2510.24012v1	Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models. Byeonghu Na, Mina Kang, Jiseok Kwak, Minsang Park, Jiwoo Shin, SeJoon Jun, Gayoung Lee, Jin-Hwa Kim, Il-Chul Moon. https://arxiv.org/abs/2510.24012
19	2510.23974v1	Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models. Byeonghu Na, Minsang Park, Gyuwon Sim, Donghyeok Shin, HeeSun Bae, Mina Kang, Se Jung Kwon, Wanmo Kang, Il-Chul Moon. https://arxiv.org/abs/2510.23974
20	2510.23776v1	Unsupervised learning for variability detection with Gaia DR3 photometry. The main sequence-white dwarf valley. P. Ranaivomanana, C. Johnston, G. Iorio, P.J. Groot, M. Uzundag, T. Kupfer, C. Aerts. https://arxiv.org/abs/2510.23776






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































