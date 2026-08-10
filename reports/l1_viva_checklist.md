# L1 Viva Checklist

## Project Flow

1. `src/preprocessing.py` loads `train.csv` and `test.csv`, cleans whitespace, maps answers `A-E` to labels `0-4`, removes duplicate MCQs, and creates a stratified 90/10 train-validation split.
2. `src/models.py` defines the custom scratch model and the pretrained DeBERTa multiple-choice model with LoRA adapters.
3. `src/train.py` trains three model families, evaluates them with accuracy, macro-F1, and MAP@3, logs runs to WandB, blends predictions, and writes `submission.csv`.
4. `src/ensemble.py` computes MAP@3, classification metrics, and searches weighted ensemble values.
5. `notebooks/Milestone3.ipynb`, `notebooks/Milestone4.ipynb`, and `notebooks/MIlestone5.ipynb` show the milestone experiments for RAG, LoRA fine-tuning, and ensembling.

## Mandatory L1 Items

### 1. Model Completion

Present these three unique models:

- Scratch model: `RegularizedMCQClassifier` in `src/models.py`, trained on TF-IDF features in `src/train.py`.
- Pretrained model: DeBERTa-v3 multiple-choice model with LoRA adapters from `get_deberta_lora_model`.
- Third model: SBERT prompt-option similarity features with Logistic Regression.

The ensemble is an additional fourth prediction strategy, not a replacement for the required three models.

### 2. WandB Tracking

`src/train.py` creates separate WandB runs:

- `Model_1_Scratch_DeepMLP`
- `Model_2_DeBERTa_LoRA`
- `Model_3_Choice_SimilarityLR`
- `Model_4_Weighted_Ensemble`

The common comparison metrics are `final_accuracy`, `final_macro_f1`, and `final_map3` for the three main models.

Before the viva, run:

```bash
wandb login
./venv/bin/python -m src.train
```

Then keep the WandB project page ready.

### 3. GitHub Authenticity

The git history shows steady progress from May 26, 2026 through July 26, 2026, with milestone commits on June 23, June 29, July 6, July 10, July 15, July 18, and July 21.

Useful command:

```bash
git log --oneline --date=short --pretty=format:'%h %ad %s' --all
```

### 4. Live Coding Check

Practice writing a small function like MAP@3:

```python
def map_at_3(top3_preds, true_labels):
    scores = []
    for preds, true in zip(top3_preds, true_labels):
        if true in preds[:3]:
            scores.append(1 / (preds.index(true) + 1))
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)
```

### 5. Code Walkthrough

Explain the flow in this order:

- Data processing: clean text, label encoding, deduplication, stratified split.
- Model setup: scratch MLP, DeBERTa+LoRA, SBERT+LogReg.
- Training: PyTorch loop for scratch model, Hugging Face `Trainer` for DeBERTa, sklearn fit for Logistic Regression.
- Evaluation: accuracy, macro-F1, MAP@3.
- Prediction: softmax probabilities, top-3 sorting, weighted ensemble, submission file.

### 6. Report Understanding

Key concepts to explain:

- RAG retrieves external context and appends it to the prompt so the model has more evidence.
- MCQ formatting turns one question into five prompt-option pairs.
- LoRA freezes most pretrained weights and trains small low-rank adapter matrices.
- Ensembling averages or weights model probabilities to improve ranking robustness.
- MAP@3 rewards the correct answer most when ranked first, less when second or third.

### 7. Authenticity Check

Be ready to explain why each model was chosen:

- Scratch MLP proves you built a model yourself.
- DeBERTa+LoRA uses pretrained language understanding efficiently.
- SBERT+LogReg is a lightweight semantic-similarity baseline.
- Ensemble combines complementary strengths from all three.
