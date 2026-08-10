import os
import gc
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass

# Import modular components from local package
from src.preprocessing import prepare_data, OPTION_COLS, INDEX_TO_OPTION
from src.models import RegularizedMCQClassifier, get_deberta_lora_model
from src.ensemble import compute_map_at_3, compute_cls_metrics, find_best_ensemble_weights

# =====================================================================
# 1. LOCAL CONFIGURATION & SPEED TOGGLE
# =====================================================================
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Set FAST_CPU_MODE = True for fast 1-minute local verification on laptop
FAST_CPU_MODE = True  

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_PATH = "./data/train.csv"
TEST_PATH  = "./data/test.csv"

MODEL_NAME  = "microsoft/deberta-v3-base"
SBERT_NAME  = "sentence-transformers/all-MiniLM-L6-v2"

# Adjust hyperparameters based on mode
if FAST_CPU_MODE:
    MAX_LENGTH  = 128   # Reduced sequence length for fast CPU speed
    BATCH_SIZE  = 8
    GRAD_ACCUM  = 2
    EPOCHS_XFMR = 1     # Single epoch for quick local check
else:
    MAX_LENGTH  = 224
    BATCH_SIZE  = 4
    GRAD_ACCUM  = 4
    EPOCHS_XFMR = 5

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_pipeline():
    print("--- Step 1: Loading & Preprocessing Data ---")
    df_train, df_val, test_df = prepare_data(
        train_path=TRAIN_PATH, 
        test_path=TEST_PATH, 
        seed=SEED
    )
    
    # If in fast local CPU mode, sample a smaller subset for instant testing
    if FAST_CPU_MODE and len(df_train) > 50:
        print("--> FAST_CPU_MODE active: Using 50 samples for rapid local execution.")
        df_train = df_train.head(50).reset_index(drop=True)
        df_val   = df_val.head(20).reset_index(drop=True)

    val_true = df_val['label'].values

    # =====================================================================
    # 2. MODEL 1: REGULARIZED PYTORCH MLP (SCRATCH)
    # =====================================================================
    print("\n--- Step 2: Training Model 1 (Regularized PyTorch MLP) ---")

    def merge_text(df):
        return df['prompt'] + " [SEP] " + df['A'] + " " + df['B'] + " " + df['C'] + " " + df['D'] + " " + df['E']

    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = torch.tensor(vectorizer.fit_transform(merge_text(df_train)).toarray(), dtype=torch.float32)
    X_val_vec   = torch.tensor(vectorizer.transform(merge_text(df_val)).toarray(), dtype=torch.float32)
    X_test_vec  = torch.tensor(vectorizer.transform(merge_text(test_df)).toarray(), dtype=torch.float32)

    y_train_scratch = torch.tensor(df_train['label'].values, dtype=torch.long)
    y_val_scratch   = torch.tensor(df_val['label'].values, dtype=torch.long)

    scratch_loader = DataLoader(TensorDataset(X_train_vec, y_train_scratch), batch_size=16, shuffle=True)

    scratch_model = RegularizedMCQClassifier(X_train_vec.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(scratch_model.parameters(), lr=5e-4, weight_decay=0.1)

    epochs_mlp = 3 if FAST_CPU_MODE else 12
    best_val_map, best_state = -1.0, None
    for epoch in range(epochs_mlp):
        scratch_model.train()
        for batch_X, batch_y in scratch_loader:
            optimizer.zero_grad()
            loss = criterion(scratch_model(batch_X.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()

        scratch_model.eval()
        with torch.no_grad():
            val_logits_epoch = scratch_model(X_val_vec.to(device)).cpu().numpy()
        val_map = compute_map_at_3(val_logits_epoch, y_val_scratch.numpy())
        if val_map > best_val_map:
            best_val_map = val_map
            best_state = {k: v.detach().clone() for k, v in scratch_model.state_dict().items()}

    if best_state is not None:
        scratch_model.load_state_dict(best_state)

    with torch.no_grad():
        scratch_val_logits = scratch_model(X_val_vec.to(device)).cpu().numpy()
        scratch_val_preds  = torch.softmax(torch.tensor(scratch_val_logits), dim=1).numpy()
        scratch_test_preds = torch.softmax(scratch_model(X_test_vec.to(device)), dim=1).cpu().numpy()

    m1_acc, m1_f1 = compute_cls_metrics(scratch_val_logits, y_val_scratch.numpy())
    m1_map = compute_map_at_3(scratch_val_logits, y_val_scratch.numpy())
    print(f"Model 1 Val -> Acc: {m1_acc:.4f} | Macro-F1: {m1_f1:.4f} | MAP@3: {m1_map:.4f}")

    del scratch_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    # 3. MODEL 2: PRETRAINED DEBERTA + LORA FINE-TUNING
    # =====================================================================
    print("\n--- Step 3: Training Model 2 (DeBERTa-v3 + LoRA) ---")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        n = len(examples['prompt'])
        first_sentences, second_sentences = [], []
        for i in range(n):
            prompt_txt = f"Context Question: {examples['prompt'][i]}"
            for opt in OPTION_COLS:
                first_sentences.append(prompt_txt)
                second_sentences.append(f"Hypothesis Option: {examples[opt][i]}")
        tokenized = tokenizer(first_sentences, second_sentences, truncation=True,
                              max_length=MAX_LENGTH, padding='max_length')
        return {k: [v[i*5:(i+1)*5] for i in range(n)] for k, v in tokenized.items()}

    def make_ds(df):
        drop_cols = [c for c in df.columns if c not in ['input_ids', 'attention_mask', 'token_type_ids', 'label']]
        return Dataset.from_pandas(df).map(preprocess_function, batched=True, batch_size=50, remove_columns=drop_cols)

    tok_train = make_ds(df_train)
    tok_val   = make_ds(df_val)
    tok_test  = make_ds(test_df)

    @dataclass
    class DataCollatorForMultipleChoice:
        tokenizer: PreTrainedTokenizerBase
        def __call__(self, features):
            has_labels = 'label' in features[0]
            labels = [int(f.pop('label')) for f in features] if has_labels else None
            bs, nc = len(features), len(features[0]['input_ids'])
            flat_features = [{k: v[i] for k, v in f.items()} for f in features for i in range(nc)]
            batch = self.tokenizer.pad(flat_features, padding=True, return_tensors='pt')
            batch = {k: v.view(bs, nc, -1) for k, v in batch.items()}
            if labels is not None:
                batch['labels'] = torch.tensor(labels, dtype=torch.long)
            return batch

    pt_model = get_deberta_lora_model(MODEL_NAME)

    args = TrainingArguments(
        output_dir='./results',
        run_name="Model_2_DeBERTa_LoRA",
        eval_strategy='epoch',
        save_strategy='no',
        learning_rate=2e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS_XFMR,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=10,
        seed=SEED
    )

    trainer = Trainer(
        model=pt_model, args=args,
        train_dataset=tok_train, eval_dataset=tok_val,
        processing_class=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer)
    )

    trainer.train()

    val_res  = trainer.predict(tok_val).predictions
    test_res = trainer.predict(tok_test).predictions
    val_logits  = val_res[0] if isinstance(val_res, tuple) else val_res
    test_logits = test_res[0] if isinstance(test_res, tuple) else test_res

    deberta_val_preds  = torch.softmax(torch.tensor(val_logits), dim=1).numpy()
    deberta_test_preds = torch.softmax(torch.tensor(test_logits), dim=1).numpy()

    m2_acc, m2_f1 = compute_cls_metrics(val_logits, val_true)
    m2_map = compute_map_at_3(val_logits, val_true)
    print(f"Model 2 Val -> Acc: {m2_acc:.4f} | Macro-F1: {m2_f1:.4f} | MAP@3: {m2_map:.4f}")

    del trainer, pt_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    # 4. MODEL 3: SBERT SIMILARITY + LOGISTIC REGRESSION
    # =====================================================================
    print("\n--- Step 4: Training Model 3 (SBERT Similarity + LogReg) ---")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)
        from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer(SBERT_NAME, device=device)

    def row_cosine_sim(a, b):
        dot = np.sum(a * b, axis=1)
        denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-8
        return dot / denom

    def build_similarity_features(df, sbert_model):
        prompt_emb = sbert_model.encode(df['prompt'].tolist(), batch_size=32, show_progress_bar=False)
        sbert_feats = []
        for opt in OPTION_COLS:
            opt_emb = sbert_model.encode(df[opt].tolist(), batch_size=32, show_progress_bar=False)
            sbert_feats.append(row_cosine_sim(prompt_emb, opt_emb))
        return np.column_stack(sbert_feats)

    X_train_sim = build_similarity_features(df_train, sbert)
    X_val_sim   = build_similarity_features(df_val, sbert)
    X_test_sim  = build_similarity_features(test_df, sbert)

    logreg = LogisticRegression(max_iter=500, C=1.0, random_state=SEED)
    logreg.fit(X_train_sim, df_train['label'].values)

    logreg_val_preds  = logreg.predict_proba(X_val_sim)
    logreg_test_preds = logreg.predict_proba(X_test_sim)

    m3_acc, m3_f1 = compute_cls_metrics(logreg_val_preds, val_true)
    m3_map = compute_map_at_3(logreg_val_preds, val_true)
    print(f"Model 3 Val -> Acc: {m3_acc:.4f} | Macro-F1: {m3_f1:.4f} | MAP@3: {m3_map:.4f}")

    del sbert
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    # 5. MODEL 4: WEIGHTED ENSEMBLE
    # =====================================================================
    print("\n--- Step 5: Finding Optimal Ensemble Weights & Exporting ---")
    best_weights, best_map = find_best_ensemble_weights(
        scratch_val_preds, deberta_val_preds, logreg_val_preds, val_true
    )

    w1, w2, w3 = best_weights
    print(f"Best Weights -> Scratch MLP: {w1:.2f}, DeBERTa+LoRA: {w2:.2f}, SBERT+LogReg: {w3:.2f}")
    print(f"Best Validation MAP@3: {best_map:.4f}")

    # Compute Test Set Predictions
    ensemble_test_probs = (w1 * scratch_test_preds) + (w2 * deberta_test_preds) + (w3 * logreg_test_preds)
    final_ranked_preds = []
    for row in ensemble_test_probs:
        top3_indices = np.argsort(-row)[:3]
        final_ranked_preds.append(' '.join([INDEX_TO_OPTION[idx] for idx in top3_indices]))

    submission = pd.DataFrame({'id': test_df['id'], 'Prediction': final_ranked_preds})
    submission.to_csv('./submission.csv', index=False)
    print("\nPipeline Execution Complete! 'submission.csv' generated successfully.")


if __name__ == "__main__":
    run_pipeline()