import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef
)

from dataset import TF66ClipDataset
from utils import build_tf66_samples, load_tf66_video_info
from model import FallTransformer


TRAIN_DIR = "../Train"
VAL_DIR = "../Validation"
EXCEL_PATH = "../Final Dataset.xlsx"

OUTPUT_DIR = "./outputs"

SEQ_LEN = 10
IMG_SIZE = 128          # En CPU, 128 es bastante más razonable que 256
BATCH_SIZE = 1          # Para Transformer en CPU, mejor empezar con 1
EPOCHS = 200
LR = 5e-5
WEIGHT_DECAY = 1e-5
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
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }


def train_one_epoch(model, loader, criterion, optimizer, device, threshold):
    model.train()

    total_loss = 0.0
    y_true, y_prob = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

        total_loss += loss.item()
        y_true.extend(y.cpu().numpy().flatten())
        y_prob.extend(probs)

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def validate(model, loader, criterion, device, threshold):
    model.eval()

    total_loss = 0.0
    y_true, y_prob = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model(x)
            loss = criterion(logits, y)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            total_loss += loss.item()
            y_true.extend(y.cpu().numpy().flatten())
            y_prob.extend(probs)

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = total_loss / len(loader)

    pos_rate = np.mean(np.array(y_prob) >= threshold)
    print(f"Val positive prediction rate: {pos_rate:.4f}")

    return metrics


def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"No se encontró el Excel: {EXCEL_PATH}")

    video_info = load_tf66_video_info(EXCEL_PATH)

    train_samples = build_tf66_samples(TRAIN_DIR, EXCEL_PATH, seq_len=SEQ_LEN)
    val_samples = build_tf66_samples(VAL_DIR, EXCEL_PATH, seq_len=SEQ_LEN)

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    if len(train_samples) == 0:
        raise ValueError("No se encontraron samples de entrenamiento.")
    if len(val_samples) == 0:
        raise ValueError("No se encontraron samples de validación.")

    labels = [int(s["label"]) for s in train_samples]
    pos = sum(labels)
    neg = len(labels) - pos

    print(f"Fall train samples: {pos}")
    print(f"NonFall train samples: {neg}")
    print(f"threshold: {THRESHOLD}")
    print(f"seq_len: {SEQ_LEN}")
    print(f"img_size: {IMG_SIZE}")

    train_ds = TF66ClipDataset(
        train_samples,
        video_info=video_info,
        img_size=IMG_SIZE,
        random_sampling=True
    )

    val_ds = TF66ClipDataset(
        val_samples,
        video_info=video_info,
        img_size=IMG_SIZE,
        random_sampling=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
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


    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_mcc = float("-inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_mcc": [],
        "val_mcc": []
    }

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, THRESHOLD
        )
        val_metrics = validate(
            model, val_loader, criterion, device, THRESHOLD
        )

        print(
            f"[{epoch}/{EPOCHS}] "
            f"Train Loss={train_metrics['loss']:.4f} "
            f"Train Acc={train_metrics['accuracy']:.4f} "
            f"Train F1={train_metrics['f1']:.4f} "
            f"Train MCC={train_metrics['mcc']:.4f} "
            f"| Val Loss={val_metrics['loss']:.4f} "
            f"Val Acc={val_metrics['accuracy']:.4f} "
            f"Val F1={val_metrics['f1']:.4f} "
            f"Val Prec={val_metrics['precision']:.4f} "
            f"Val Rec={val_metrics['recall']:.4f} "
            f"Val MCC={val_metrics['mcc']:.4f}"
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])
        history["train_mcc"].append(train_metrics["mcc"])
        history["val_mcc"].append(val_metrics["mcc"])

        if val_metrics["mcc"] > best_mcc:
            best_mcc = val_metrics["mcc"]
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print("**Mejor modelo guardado por MCC**")

    with open(os.path.join(OUTPUT_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print("\nEntrenamiento finalizado")
    print(f"Mejor MCC: {best_mcc:.4f}")


if __name__ == "__main__":
    main()