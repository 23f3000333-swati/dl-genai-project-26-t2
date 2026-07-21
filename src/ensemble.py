import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_map_at_3(logits: np.ndarray, labels: np.ndarray) -> float:
    """Calculates Mean Average Precision @ 3."""
    top3 = np.argsort(-logits, axis=1)[:, :3]
    scores = []
    for i, true_label in enumerate(labels):
        if true_label == top3[i, 0]:
            scores.append(1.0)
        elif true_label == top3[i, 1]:
            scores.append(0.5)
        elif true_label == top3[i, 2]:
            scores.append(1 / 3)
        else:
            scores.append(0.0)
    return float(np.mean(scores))

def compute_cls_metrics(logits: np.ndarray, labels: np.ndarray):
    """Calculates Accuracy and Macro F1 score."""
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average='macro')
    return acc, f1m

def find_best_ensemble_weights(preds_m1, preds_m2, preds_m3, val_labels, step=0.05):
    """Grid searches optimal blending weights on validation set."""
    best_map = -1.0
    best_weights = (0.33, 0.33, 0.34)

    for w1 in np.arange(0, 1.0001, step):
        for w2 in np.arange(0, 1.0001 - w1, step):
            w3 = max(0.0, 1.0 - w1 - w2)
            blend = w1 * preds_m1 + w2 * preds_m2 + w3 * preds_m3
            score = compute_map_at_3(blend, val_labels)
            if score > best_map:
                best_map = score
                best_weights = (w1, w2, w3)

    return best_weights, best_map