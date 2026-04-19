import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef

from dataset import TF66ClipDataset
from utils import build_tf66_samples, load_tf66_video_info
from model import FallNet


VAL_DIR = "../Validation"
EXCEL_PATH = "../Final Dataset.xlsx"
MODEL_PATH = "./outputs/best_model.pth"

SEQ_LEN = 10
IMG_SIZE = 128
BATCH_SIZE = 2
THRESHOLD = 0.5


def compute_metrics(y_true, y_prob):
    y_pred = [1 if p >= THRESHOLD else 0 for p in y_prob]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def main():
    device = "cpu"
    print(f"Usando dispositivo: {device}")

    video_info = load_tf66_video_info(EXCEL_PATH)
    val_samples = build_tf66_samples(VAL_DIR, video_info, SEQ_LEN)

    val_ds = TF66ClipDataset(val_samples, img_size=IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FallNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true, y_prob = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            y_true.extend(y.numpy().flatten())
            y_prob.extend(probs)

    metrics = compute_metrics(y_true, y_prob)

    print("\n=== RESULTADOS ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"MCC      : {metrics['mcc']:.4f}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    with open("./outputs/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()