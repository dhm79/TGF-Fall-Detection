import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

from dataset import TF66ClipDataset
from utils import build_tf66_samples, load_tf66_video_info
from model import FallNet


# =========================
# CONFIG
# =========================
TRAIN_DIR = "../Train"
VAL_DIR = "../Validation"
EXCEL_PATH = "../Final Dataset.xlsx"

OUTPUT_DIR = "./outputs"

SEQ_LEN = 10
IMG_SIZE = 128
BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
THRESHOLD = 0.5


# =========================
# UTILS
# =========================
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
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }


# =========================
# TRAIN / VAL
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    y_true, y_prob = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

        total_loss += loss.item()
        y_true.extend(y.cpu().numpy().flatten())
        y_prob.extend(probs)

    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    y_true, y_prob = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)

            logits = model(x)
            loss = criterion(logits, y)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            total_loss += loss.item()
            y_true.extend(y.cpu().numpy().flatten())
            y_prob.extend(probs)

    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# =========================
# MAIN
# =========================
def main():
    set_seed()
    ensure_dir(OUTPUT_DIR)

    device = "cpu"
    print(f"Usando dispositivo: {device}")

    # -------- LOAD EXCEL --------
    video_info = load_tf66_video_info(EXCEL_PATH)

    # -------- BUILD DATA --------
    train_samples = build_tf66_samples(TRAIN_DIR, video_info, SEQ_LEN)
    val_samples = build_tf66_samples(VAL_DIR, video_info, SEQ_LEN)

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # -------- DATASET --------
    train_ds = TF66ClipDataset(train_samples, img_size=IMG_SIZE)
    val_ds = TF66ClipDataset(val_samples, img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # -------- MODEL --------
    model = FallNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_mcc = -1

    for epoch in range(1, EPOCHS + 1):
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate(model, val_loader, criterion, device)

        print(
            f"[{epoch}/{EPOCHS}] "
            f"Train Loss={train_m['loss']:.4f} "
            f"Train MCC={train_m['mcc']:.4f} | "
            f"Val Loss={val_m['loss']:.4f} "
            f"Val MCC={val_m['mcc']:.4f}"
        )

        if val_m["mcc"] > best_mcc:
            best_mcc = val_m["mcc"]
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print("** Mejor modelo guardado **")

    print(f"\nMejor MCC: {best_mcc:.4f}")


if __name__ == "__main__":
    main()