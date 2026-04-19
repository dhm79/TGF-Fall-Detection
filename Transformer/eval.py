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
from utils import build_tf66_samples, load_tf66_video_info
from model import FallTransformer


VAL_DIR = "../Validation"
EXCEL_PATH = "../Final Dataset.xlsx"
MODEL_PATH = "./outputs/best_model.pth"
OUTPUT_DIR = "./outputs"

SEQ_LEN = 10
IMG_SIZE = 128
BATCH_SIZE = 1
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"No se encontró el Excel: {EXCEL_PATH}")

    video_info = load_tf66_video_info(EXCEL_PATH)

    val_samples = build_tf66_samples(VAL_DIR, EXCEL_PATH, seq_len=SEQ_LEN)
    if len(val_samples) == 0:
        raise ValueError("No se encontraron samples en Validation.")

    print(f"Validation samples: {len(val_samples)}")

    val_ds = TF66ClipDataset(
        val_samples,
        video_info=video_info,
        img_size=IMG_SIZE,
        random_sampling=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = FallTransformer(
        img_size=IMG_SIZE,
        patch_size=16,
        seq_len=SEQ_LEN,
        embed_dim=64,
        spatial_depth=1,
        temporal_depth=1,
        num_heads=4,
        dropout=0.1
    ).to(device)

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

    print("\n=== RESULTADOS EN VALIDATION ===")
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