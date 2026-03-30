import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef
)

from dataset import TF66ClipDataset
from utils import build_tf66_clips
from model import FallNet


VAL_DIR = "../Validation"
MODEL_PATH = "./outputs/best_model.pth"
OUTPUT_DIR = "./outputs"

SEQ_LEN = 10
STRIDE = 20
IMG_SIZE = 96
BATCH_SIZE = 2
NUM_WORKERS = 0
SEED = 42
THRESHOLD = 0.5


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    device = "cpu"
    print(f"Usando dispositivo: {device}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    val_clips = build_tf66_clips(VAL_DIR, seq_len=SEQ_LEN, stride=STRIDE)
    if len(val_clips) == 0:
        raise ValueError("No se encontraron clips en Validation.")

    print(f"Validation clips: {len(val_clips)}")

    val_ds = TF66ClipDataset(val_clips, img_size=IMG_SIZE)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = FallNet(hidden_size=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            y_true.extend(y.numpy().flatten())
            y_prob.extend(probs)

    metrics = compute_metrics(y_true, y_prob, threshold=THRESHOLD)

    print("\n=== RESULTADOS EN VALIDATION (CLIPS) ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"MCC      : {metrics['mcc']:.4f}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMétricas guardadas en: {os.path.join(OUTPUT_DIR, 'eval_metrics.json')}")


if __name__ == "__main__":
    main()