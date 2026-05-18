"""
Entrenamiento del Transformer espacio-temporal para TF-66

Ejecutar desde la carpeta del Transformer:

    python train.py
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import json
import copy
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
    matthews_corrcoef,
)

from dataset import TF66ClipDataset
from utils import build_tf66_samples, load_tf66_video_info
from model import FallTransformer


# ============================================================
# RUTAS
# ============================================================

TRAIN_DIR = "../Train"
VAL_DIR = "../Validation"
EXCEL_PATH = "../Final Dataset.xlsx"

OUTPUT_DIR = "./outputs"


# ============================================================
# HIPERPARÁMETROS
# ============================================================

SEQ_LEN = 10
IMG_SIZE = 256

BATCH_SIZE = 4
EPOCHS = 200

LR = 5e-5

# Más regularización para reducir sobreajuste
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0
SEED = 42
THRESHOLD = 0.5

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def set_seed(seed=42):
    """
    Fija semillas para mejorar la reproducibilidad.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    """
    Crea una carpeta si no existe.
    """
    os.makedirs(path, exist_ok=True)


def save_json(data, path):
    """
    Guarda un diccionario en formato JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Convierte probabilidades en predicciones binarias y calcula métricas.
    """
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


# ============================================================
# ENTRENAMIENTO DE UNA ÉPOCA
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, threshold):
    """
    Entrena el modelo durante una época.
    """
    model.train()

    total_loss = 0.0
    y_true = []
    y_prob = []

    for x, y in loader:
        x = x.to(device)
        y = y.float().to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()

        # Ayuda a evitar gradientes demasiado grandes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

        total_loss += loss.item()
        y_true.extend(y.detach().cpu().numpy().flatten())
        y_prob.extend(probs)

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = total_loss / max(len(loader), 1)

    return metrics


# ============================================================
# VALIDACIÓN
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device, threshold):
    """
    Evalúa el modelo en validación.
    """
    model.eval()

    total_loss = 0.0
    y_true = []
    y_prob = []

    for x, y in loader:
        x = x.to(device)
        y = y.float().to(device).unsqueeze(1)

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

        total_loss += loss.item()
        y_true.extend(y.detach().cpu().numpy().flatten())
        y_prob.extend(probs)

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = total_loss / max(len(loader), 1)

    pos_rate = float(np.mean(np.array(y_prob) >= threshold))
    metrics["positive_prediction_rate"] = pos_rate

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # --------------------------------------------------------
    # 1) Comprobar rutas
    # --------------------------------------------------------
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"No se encontró el Excel: {EXCEL_PATH}")

    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"No se encontró Train: {TRAIN_DIR}")

    if not os.path.exists(VAL_DIR):
        raise FileNotFoundError(f"No se encontró Validation: {VAL_DIR}")

    # --------------------------------------------------------
    # 2) Cargar metadata temporal
    # --------------------------------------------------------
    video_info = load_tf66_video_info(EXCEL_PATH)

    # --------------------------------------------------------
    # 3) Construir muestras
    # --------------------------------------------------------
    train_samples = build_tf66_samples(TRAIN_DIR, EXCEL_PATH, seq_len=SEQ_LEN)
    val_samples = build_tf66_samples(VAL_DIR, EXCEL_PATH, seq_len=SEQ_LEN)

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples  : {len(val_samples)}")

    if len(train_samples) == 0:
        raise ValueError("No se encontraron samples de entrenamiento.")

    if len(val_samples) == 0:
        raise ValueError("No se encontraron samples de validación.")

    labels = [int(s["label"]) for s in train_samples]
    pos = sum(labels)
    neg = len(labels) - pos

    print(f"Fall train samples   : {pos}")
    print(f"NonFall train samples: {neg}")
    print(f"Threshold            : {THRESHOLD}")
    print(f"Seq len              : {SEQ_LEN}")
    print(f"Img size             : {IMG_SIZE}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Epochs máximas       : {EPOCHS}")
    print(f"LR                   : {LR}")
    print(f"Weight decay         : {WEIGHT_DECAY}")

    # --------------------------------------------------------
    # 4) Datasets
    # --------------------------------------------------------
    train_ds = TF66ClipDataset(
        train_samples,
        video_info=video_info,
        img_size=IMG_SIZE,
        random_sampling=True,
    )

    val_ds = TF66ClipDataset(
        val_samples,
        video_info=video_info,
        img_size=IMG_SIZE,
        random_sampling=False,
    )

    # --------------------------------------------------------
    # 5) DataLoaders
    # --------------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # 6) Modelo
    # --------------------------------------------------------
    model = FallTransformer(
        img_size=IMG_SIZE,
        patch_size=16,
        seq_len=SEQ_LEN,
        embed_dim=96,
        spatial_depth=1,
        temporal_depth=1,
        num_heads=4,
        dropout=0.2,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # 7) Variables de seguimiento
    # --------------------------------------------------------
    best_mcc = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

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
        "val_mcc": [],
        "val_positive_prediction_rate": [],
    }

    train_config = {
        "seq_len": SEQ_LEN,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "threshold": THRESHOLD,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_fall_samples": pos,
        "train_nonfall_samples": neg,
        "device": device,
        "model": {
            "name": "FallTransformer",
            "patch_size": 16,
            "embed_dim": 96,
            "spatial_depth": 1,
            "temporal_depth": 1,
            "num_heads": 4,
            "dropout": 0.2,
        },
    }

    save_json(train_config, os.path.join(OUTPUT_DIR, "train_config.json"))

    # --------------------------------------------------------
    # 8) Bucle de entrenamiento
    # --------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=THRESHOLD,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=THRESHOLD,
        )

        print(
            f"[{epoch:03d}/{EPOCHS}] "
            f"Train Loss={train_metrics['loss']:.4f} "
            f"Train Acc={train_metrics['accuracy']:.4f} "
            f"Train F1={train_metrics['f1']:.4f} "
            f"Train MCC={train_metrics['mcc']:.4f} "
            f"| Val Loss={val_metrics['loss']:.4f} "
            f"Val Acc={val_metrics['accuracy']:.4f} "
            f"Val F1={val_metrics['f1']:.4f} "
            f"Val Prec={val_metrics['precision']:.4f} "
            f"Val Rec={val_metrics['recall']:.4f} "
            f"Val MCC={val_metrics['mcc']:.4f} "
            f"Val PosRate={val_metrics['positive_prediction_rate']:.4f}"
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
        history["val_positive_prediction_rate"].append(
            val_metrics["positive_prediction_rate"]
        )

        save_json(history, os.path.join(OUTPUT_DIR, "history.json"))

        current_mcc = val_metrics["mcc"]
        current_val_loss = val_metrics["loss"]

        if current_mcc > best_mcc:
            best_mcc = current_mcc
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

            torch.save(best_model_state, os.path.join(OUTPUT_DIR, "best_model.pth"))

            print(
                f"** Mejor modelo guardado | "
                f"Epoch={best_epoch} | "
                f"Val MCC={best_mcc:.4f} | "
                f"Val Loss={best_val_loss:.4f} **"
            )

    # --------------------------------------------------------
    # 9) Restaurar y guardar el mejor modelo final
    # --------------------------------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))

    save_json(history, os.path.join(OUTPUT_DIR, "history.json"))

    summary = {
        "best_epoch": best_epoch,
        "best_val_mcc": best_mcc,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
        "model_path": os.path.join(OUTPUT_DIR, "best_model.pth"),
        "history_path": os.path.join(OUTPUT_DIR, "history.json"),
    }

    save_json(summary, os.path.join(OUTPUT_DIR, "training_summary.json"))

    print("\nEntrenamiento finalizado")
    print(f"Épocas entrenadas : {len(history['train_loss'])}")
    print(f"Mejor época       : {best_epoch}")
    print(f"Mejor Val MCC     : {best_mcc:.4f}")
    print(f"Val Loss mejor MCC: {best_val_loss:.4f}")
    print(f"Modelo guardado en: {os.path.join(OUTPUT_DIR, 'best_model.pth')}")
    print(f"Histórico guardado: {os.path.join(OUTPUT_DIR, 'history.json')}")
    print(f"Resumen guardado  : {os.path.join(OUTPUT_DIR, 'training_summary.json')}")


if __name__ == "__main__":
    main()