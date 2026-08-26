# dl-genai-project-26-t2
# Smart MCQ Solver Challenge 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dl-genai-project-26-t2-jbmmn9qxefzcaldu3sydbk.streamlit.app/)

**Author:** Swati Singh  
**Roll No. / Email:** [23f3000333@ds.study.iitm.ac.in](mailto:23f3000333@ds.study.iitm.ac.in)  
**Live Application:** [Smart MCQ Solver AI App](https://dl-genai-project-26-t2-jbmmn9qxefzcaldu3sydbk.streamlit.app/)

---

##  Project Overview

The **Smart MCQ Solver** is a natural language processing (NLP) and deep learning pipeline designed to predict the correct answer for 5-option multiple-choice questions ($A, B, C, D, E$). 

By combining classical text feature extraction with modern transformer fine-tuning and semantic similarity search, this project uses a **weighted ensemble architecture** to output ranked option probabilities and MAP@3 metrics.

---

## Live Demo

Access the interactive web interface deployed on Streamlit Cloud:
**[Launch Smart MCQ Solver AI](https://dl-genai-project-26-t2-jbmmn9qxefzcaldu3sydbk.streamlit.app/)**

---

## Model Architecture & Pipeline

The pipeline integrates three distinct model paradigms into an ensemble:

1. **Model 1: Regularized PyTorch MLP (From Scratch)**
   * Extracts text features using TF-IDF n-grams ($1, 2$).
   * Passes sparse vector representations into a multi-layer perceptron (MLP) built with PyTorch, utilizing Dropout, BatchNorm, and Weight Decay regularization.

2. **Model 2: DeBERTa-v3 + LoRA Fine-Tuning**
   * Uses `microsoft/deberta-v3-base` pre-trained transformer architecture.
   * Fine-tunes the multiple-choice head efficiently using **LoRA (Low-Rank Adaptation)** via Hugging Face `peft`.

3. **Model 3: SBERT Cosine Similarity + Logistic Regression**
   * Computes semantic embedding vectors for prompts and individual options using `sentence-transformers/all-MiniLM-L6-v2`.
   * Evaluates pairwise cosine similarities and trains a Logistic Regression classifier on similarity scores.

4. **Model 4: Weighted Ensemble Blending**
   * Blends predicted probability distributions across all models using grid-searched optimal validation weights to optimize MAP@3.

---

## Repository Layout

```text
dl-genai-project-26-t2/
├── data/                  # Local training and test CSV files (git-ignored)
├── src/                   # Modular Python package
│   ├── __init__.py
│   ├── preprocessing.py   # Dataset loading, cleaning, and stratification
│   ├── models.py          # PyTorch MLP & DeBERTa LoRA architecture definitions
│   ├── ensemble.py        # Evaluation metrics (MAP@3, F1) & ensemble optimization
│   └── train.py           # Main training execution script
├── app.py                 # Streamlit deployment web application
├── requirements.txt       # Project dependencies
├── .gitignore             # Excludes large binaries (venv, models, wandb)
└── README.md              # Project documentation
```

## Local Installation & Execution

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone [https://github.com/23f3000333-swati/dl-genai-project-26-t2.git](https://github.com/23f3000333-swati/dl-genai-project-26-t2.git)
cd dl-genai-project-26-t2

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```
