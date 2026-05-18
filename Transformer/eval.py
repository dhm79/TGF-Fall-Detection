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
    matthews_corrcoef,
)

from dataset import TF66ClipDataset
from utils import build_tf66_samples, load_tf66_video_info
from model import FallTransformer


VAL_DIR = "../Validation"
EXCEL_PATH = "../Final Dataset.xlsx"
MODEL_PATH = "./outputs/best_model.pth"
OUTPUT_DIR = "./outputs"

SEQ_LEN = 10
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_WORKERS = 0
SEED = 42
THRESHOLD = 0.5

# Debe coincidir con train.py
PATCH_SIZE = 16
EMBED_DIM = 96
SPATIAL_DEPTH = 1
TEMPORAL_DEPTH = 1
NUM_HEADS = 4
DROPOUT = 0.2


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "positive_prediction_rate": float(np.mean(y_pred)),
    }


def load_model_weights(model, model_path, device):
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    return model

def evaluate_thresholds(y_true, y_prob):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob, dtype=np.float32)

    thresholds = [round(x, 2) for x in np.arange(0.30, 0.99, 0.05)]

    results = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "positive_prediction_rate": float(np.mean(y_pred)),
        })

    return results

def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"No se encontró el Excel: {EXCEL_PATH}")

    if not os.path.exists(VAL_DIR):
        raise FileNotFoundError(f"No se encontró Validation: {VAL_DIR}")

    video_info = load_tf66_video_info(EXCEL_PATH)

    val_samples = build_tf66_samples(VAL_DIR, EXCEL_PATH, seq_len=SEQ_LEN)

    if len(val_samples) == 0:
        raise ValueError("No se encontraron samples en Validation.")

    print(f"Validation samples: {len(val_samples)}")
    print(f"Modelo cargado desde: {MODEL_PATH}")
    print(f"Seq len  : {SEQ_LEN}")
    print(f"Img size : {IMG_SIZE}")
    print(f"Threshold: {THRESHOLD}")

    val_ds = TF66ClipDataset(
        val_samples,
        video_info=video_info,
        img_size=IMG_SIZE,
        random_sampling=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = FallTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        spatial_depth=SPATIAL_DEPTH,
        temporal_depth=TEMPORAL_DEPTH,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(device)

    model = load_model_weights(model, MODEL_PATH, device)
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

            y_true.extend(y.detach().cpu().numpy().flatten())
            y_prob.extend(probs)

    metrics = compute_metrics(y_true, y_prob, threshold=THRESHOLD)

    threshold_results = evaluate_thresholds(y_true, y_prob)

    print("\n=== RESULTADOS POR THRESHOLD ===")
    for r in threshold_results:
        print(
            f"thr={r['threshold']:.2f} | "
            f"recall={r['recall']:.4f} | "
            f"precision={r['precision']:.4f} | "
            f"f1={r['f1_score']:.4f} | "
            f"mcc={r['mcc']:.4f} | "
            f"TP={r['tp']} | FP={r['fp']} | "
            f"TN={r['tn']} | FN={r['fn']}"
        )

    save_json(threshold_results, os.path.join(OUTPUT_DIR, "threshold_results.json"))

    metrics["threshold"] = THRESHOLD
    metrics["seq_len"] = SEQ_LEN
    metrics["img_size"] = IMG_SIZE
    metrics["batch_size"] = BATCH_SIZE
    metrics["num_validation_samples"] = len(val_samples)
    metrics["model_path"] = MODEL_PATH
    metrics["model"] = {
        "name": "FallTransformer",
        "patch_size": PATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "spatial_depth": SPATIAL_DEPTH,
        "temporal_depth": TEMPORAL_DEPTH,
        "num_heads": NUM_HEADS,
        "dropout": DROPOUT,
    }

    print("\n=== RESULTADOS EN VALIDATION ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"MCC      : {metrics['mcc']:.4f}")
    print(f"PosRate  : {metrics['positive_prediction_rate']:.4f}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    save_json(metrics, os.path.join(OUTPUT_DIR, "eval_metrics.json"))

    print(f"\nMétricas guardadas en: {os.path.join(OUTPUT_DIR, 'eval_metrics.json')}")


if __name__ == "__main__":
    main()