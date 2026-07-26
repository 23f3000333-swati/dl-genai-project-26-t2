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
import wandb

# Import modular components from local package
from src.preprocessing import prepare_data, OPTION_COLS, INDEX_TO_OPTION
from src.models import RegularizedMCQClassifier, get_deberta_lora_model
from src.ensemble import compute_map_at_3, compute_cls_metrics, find_best_ensemble_weights

# =====================================================================
# 1. CONFIGURATION & SEEDS
# =====================================================================
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_PROJECT"] = "smart-mcq-solver-challenge"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_NAME  = "microsoft/deberta-v3-base"
SBERT_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH  = 224
BATCH_SIZE  = 4
GRAD_ACCUM  = 4
EPOCHS_XFMR = 5

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_pipeline():
    print("--- Step 1: Loading & Preprocessing Data ---")
    df_train, df_val, test_df = prepare_data(
        train_path="./data/train.csv", 
        test_path="./data/test.csv", 
        seed=SEED
    )
    
    val_true = df_val['label'].values

    # =====================================================================
    # 2. MODEL 1: REGULARIZED PYTORCH MLP (SCRATCH)
    # =====================================================================
    print("\n--- Step 2: Training Model 1 (Regularized PyTorch MLP) ---")
    wandb.init(project="smart-mcq-solver-challenge", name="Model_1_Scratch_DeepMLP", job_type="train", reinit=True)

    def merge_text(df):
        return df['prompt'] + " [SEP] " + df['A'] + " " + df['B'] + " " + df['C'] + " " + df['D'] + " " + df['E']

    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = torch.tensor(vectorizer.fit_transform(merge_text(df_train)).toarray(), dtype=torch.float32)
    X_val_vec   = torch.tensor(vectorizer.transform(merge_text(df_val)).toarray(), dtype=torch.float32)
    X_test_vec  = torch.tensor(vectorizer.transform(merge_text(test_df)).toarray(), dtype=torch.float32)

    y_train_scratch = torch.tensor(df_train['label'].values, dtype=torch.long)
    y_val_scratch   = torch.tensor(df_val['label'].values, dtype=torch.long)

    scratch_loader = DataLoader(TensorDataset(X_train_vec, y_train_scratch), batch_size=32, shuffle=True)

    scratch_model = RegularizedMCQClassifier(X_train_vec.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(scratch_model.parameters(), lr=5e-4, weight_decay=0.1)

    best_val_map, best_state = -1.0, None
    for epoch in range(12):
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
    
    wandb.log({"final_accuracy": m1_acc, "final_macro_f1": m1_f1, "final_map3": m1_map})
    wandb.finish()

    del scratch_model
    gc.collect(); torch.cuda.empty_cache()

    # =====================================================================
    # 3. MODEL 2: PRETRAINED DEBERTA + LORA FINE-TUNING
    # =====================================================================
    print("\n--- Step 3: Training Model 2 (DeBERTa-v3 + LoRA) ---")
    wandb.init(project="smart-mcq-solver-challenge", name="Model_2_DeBERTa_LoRA", job_type="train", reinit=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

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
        return Dataset.from_pandas(df).map(preprocess_function, batched=True, batch_size=100, remove_columns=drop_cols)

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
    pt_model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir='./results',
        run_name="Model_2_DeBERTa_LoRA",
        eval_strategy='epoch',
        save_strategy='no',
        learning_rate=2e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS_XFMR,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        logging_steps=25,
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
    
    wandb.finish()

    del trainer, pt_model
    gc.collect(); torch.cuda.empty_cache()

    # =====================================================================
    # 4. MODEL 3: SBERT SIMILARITY + LOGISTIC REGRESSION
    # =====================================================================
    print("\n--- Step 4: Training Model 3 (SBERT Similarity + LogReg) ---")
    wandb.init(project="smart-mcq-solver-challenge", name="Model_3_Choice_SimilarityLR", job_type="train", reinit=True)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)
        from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer(SBERT_NAME, device=device)

    def row_cosine_sim(a, b):
        dot = np.sum(a * b, axis=1)
        denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-8
        return dot / denom

    def build_similarity_features(df, sbert_model):
        prompt_emb = sbert_model.encode(df['prompt'].tolist(), batch_size=64, show_progress_bar=False)
        sbert_feats = []
        for opt in OPTION_COLS:
            opt_emb = sbert_model.encode(df[opt].tolist(), batch_size=64, show_progress_bar=False)
            sbert_feats.append(row_cosine_sim(prompt_emb, opt_emb))
        return np.column_stack(sbert_feats)

    X_train_sim = build_similarity_features(df_train, sbert)
    X_val_sim   = build_similarity_features(df_val, sbert)
    X_test_sim  = build_similarity_features(test_df, sbert)

    logreg = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    logreg.fit(X_train_sim, df_train['label'].values)

    logreg_val_preds  = logreg.predict_proba(X_val_sim)
    logreg_test_preds = logreg.predict_proba(X_test_sim)

    m3_acc, m3_f1 = compute_cls_metrics(logreg_val_preds, val_true)
    m3_map = compute_map_at_3(logreg_val_preds, val_true)
    print(f"Model 3 Val -> Acc: {m3_acc:.4f} | Macro-F1: {m3_f1:.4f} | MAP@3: {m3_map:.4f}")
    
    wandb.log({"final_accuracy": m3_acc, "final_macro_f1": m3_f1, "final_map3": m3_map})
    wandb.finish()

    del sbert
    gc.collect(); torch.cuda.empty_cache()

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

    wandb.init(project="smart-mcq-solver-challenge", name="Model_4_Weighted_Ensemble", job_type="ensemble", reinit=True)
    wandb.log({"ensemble_map3": best_map, "w_scratch": w1, "w_deberta": w2, "w_logreg": w3})
    wandb.finish()

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