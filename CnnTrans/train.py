import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef
)

from dataset import TF66ClipDataset
from utils import build_tf66_clips
from model_transformer import TransformerFallNet


# =========================
# CONFIGURACIÓN
# =========================
TRAIN_DIR = "../Train"
VAL_DIR = "../Validation"

OUTPUT_DIR = "./outputs"

SEQ_LEN = 10
STRIDE = 20          # antes 10; subido para que no genere tantos clips
IMG_SIZE = 96
BATCH_SIZE = 2
EPOCHS = 2
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0
SEED = 42
THRESHOLD = 0.5


# =========================
# UTILIDADES
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
    return metrics


# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    device = "cpu"
    print(f"Usando dispositivo: {device}")

    # -------------------------
    # Generar clips
    # -------------------------
    train_clips = build_tf66_clips(TRAIN_DIR, seq_len=SEQ_LEN, stride=STRIDE)
    val_clips = build_tf66_clips(VAL_DIR, seq_len=SEQ_LEN, stride=STRIDE)

    print(f"Train clips: {len(train_clips)}")
    print(f"Val clips: {len(val_clips)}")

    if len(train_clips) == 0:
        raise ValueError("No se encontraron clips de entrenamiento.")
    if len(val_clips) == 0:
        raise ValueError("No se encontraron clips de validación.")

    # -------------------------
    # Balance de clases en clips
    # -------------------------
    labels = [int(c["label"]) for c in train_clips]
    pos = sum(labels)                 # Fall = 1
    neg = len(labels) - pos           # NonFall = 0

    print(f"Fall train clips: {pos}")
    print(f"NonFall train clips: {neg}")
    print(f"threshold: {THRESHOLD}")
    print(f"stride: {STRIDE}")

    # -------------------------
    # Dataset
    # -------------------------
    train_ds = TF66ClipDataset(train_clips, img_size=IMG_SIZE)
    val_ds = TF66ClipDataset(val_clips, img_size=IMG_SIZE)

    # -------------------------
    # Sampler balanceado
    # -------------------------
    class_counts = np.array([neg, pos], dtype=np.float32)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # -------------------------
    # Modelo
    # -------------------------


        # PARA PC CON GPU  
    # 
    # TransformerFallNet(
    #   feature_dim=256,
    #   num_heads=8,
    #   num_layers=4,
    #   ff_dim=512,
    #   dropout=0.3
    # )
    #
    
    model = TransformerFallNet(
        feature_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.3
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_mcc = -1.0

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