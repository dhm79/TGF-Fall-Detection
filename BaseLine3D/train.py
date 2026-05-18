"""
Entrenamiento del baseline 3D CNN con balanceo de clases y early stopping.

1. Usa rutas relativas:
   - ../Train
   - ../Validation
   - ../Final Dataset.xlsx

2. Lee automáticamente el Excel original de TF-66.

3. Construye las muestras de entrenamiento y validación
   a partir de las carpetas de frames.

4. Usa el Dataset.xlsx para seleccionar clips:
   - Fall: guiado por el Excel
   - NonFall: ventana válida aleatoria

5. Aplica balanceo en entrenamiento con WeightedRandomSampler.

6. Entrena el modelo 3D CNN.

7. Aplica early stopping según MCC de validación.

8. Guarda:
   - outputs/best_model.pth
   - outputs/history.json
   - outputs/train_config.json

Cómo ejecutarlo
---------------
Desde la carpeta BaseLine3D:

    python train.py
    python eval.py
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import json
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

from dataset import TF66ClipDataset
from utils import build_tf66_samples, load_tf66_video_info
from model import TF66Baseline3DCNN


# ============================================================
# RUTAS RELATIVAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TF66_ROOT = BASE_DIR.parent

TRAIN_DIR = TF66_ROOT / "Train"
VAL_DIR = TF66_ROOT / "Validation"
EXCEL_PATH = TF66_ROOT / "Final Dataset.xlsx"

OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================
# HIPERPARÁMETROS
# ============================================================

SEQ_LEN = 10
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4

# Regularización para reducir sobreajuste
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0
SEED = 42
THRESHOLD = 0.5

# Early stopping
EARLY_STOPPING = True
PATIENCE = 15
MIN_DELTA = 0.001
MONITOR = "mcc" 


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def set_seed(seed=42):
    """
    Fija semillas para que los resultados sean más reproducibles.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Mejora reproducibilidad, aunque puede reducir algo el rendimiento
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
    y_prob = np.array(y_prob)
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

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    threshold: float,
) -> dict:
    """
    Entrena el modelo durante una época completa.
    """
    model.train()

    total_loss = 0.0
    y_true = []
    y_prob = []

    for x, y in loader:
        x = x.to(device)
        y = y.float().to(device).unsqueeze(1)  # -> (B, 1)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
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
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    threshold: float,
) -> dict:
    """
    Evalúa el modelo en validación sin actualizar pesos.
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
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el Excel: {EXCEL_PATH}")

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"No se encontró la carpeta Train: {TRAIN_DIR}")

    if not VAL_DIR.exists():
        raise FileNotFoundError(f"No se encontró la carpeta Validation: {VAL_DIR}")

    # --------------------------------------------------------
    # 2) Cargar metadata temporal del Excel
    # --------------------------------------------------------
    video_info = load_tf66_video_info(EXCEL_PATH)

    # --------------------------------------------------------
    # 3) Construir muestras a nivel de vídeo
    # --------------------------------------------------------
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

    if pos == 0 or neg == 0:
        raise ValueError(
            f"El conjunto de entrenamiento debe tener ambas clases. "
            f"Fall={pos}, NonFall={neg}"
        )

    print(f"Fall train samples   : {pos}")
    print(f"NonFall train samples: {neg}")
    print(f"Threshold            : {THRESHOLD}")
    print(f"Seq len              : {SEQ_LEN}")
    print(f"Img size             : {IMG_SIZE}")
    print(f"Batch                : {BATCH_SIZE}")
    print(f"Epochs máximas       : {EPOCHS}")
    print(f"LR                   : {LR}")
    print(f"Weight decay         : {WEIGHT_DECAY}")
    print(f"Early stopping       : {EARLY_STOPPING}")
    print(f"Patience             : {PATIENCE}")
    print(f"Min delta            : {MIN_DELTA}")
    print(f"Monitor              : val_{MONITOR}")

    # --------------------------------------------------------
    # 4) Crear Dataset
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
    # 5) Balanceo de clases en entrenamiento
    # --------------------------------------------------------
    sample_weights = []

    for s in train_samples:
        label = int(s["label"])

        if label == 1:
            sample_weights.append(1.0 / pos)
        else:
            sample_weights.append(1.0 / neg)

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # --------------------------------------------------------
    # 6) DataLoaders
    # --------------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
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
    # 7) Crear modelo, pérdida y optimizador
    # --------------------------------------------------------
    model = TF66Baseline3DCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # 8) Variables de seguimiento
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
        "early_stopping": EARLY_STOPPING,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "monitor": MONITOR,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_fall_samples": pos,
        "train_nonfall_samples": neg,
        "device": device,
    }

    save_json(train_config, OUTPUT_DIR / "train_config.json")

    # --------------------------------------------------------
    # 9) Bucle de entrenamiento
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
            f"Val MCC={val_metrics['mcc']:.4f}"
        )

        # Guardar histórico
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

        # ----------------------------------------------------
        # 10) Guardado del mejor modelo por MCC
        # ----------------------------------------------------
        current_mcc = val_metrics["mcc"]
        current_val_loss = val_metrics["loss"]

        improved = current_mcc > best_mcc + MIN_DELTA

        if improved:
            best_mcc = current_mcc
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save(best_model_state, OUTPUT_DIR / "best_model.pth")

            print(
                f"** Mejor modelo guardado | "
                f"Epoch={best_epoch} | "
                f"Val MCC={best_mcc:.4f} | "
                f"Val Loss={best_val_loss:.4f} **"
            )

        else:
            epochs_without_improvement += 1

            print(
                f"Sin mejora en Val MCC durante "
                f"{epochs_without_improvement}/{PATIENCE} épocas"
            )

        # Guardar histórico en cada época por seguridad
        save_json(history, OUTPUT_DIR / "history.json")

        # ----------------------------------------------------
        # 11) Early stopping
        # ----------------------------------------------------
        if EARLY_STOPPING and epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping activado.")
            print(
                f"No hubo mejora de Val MCC superior a {MIN_DELTA} "
                f"durante {PATIENCE} épocas."
            )
            break

    # --------------------------------------------------------
    # 12) Restaurar el mejor modelo al final
    # --------------------------------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")

    # --------------------------------------------------------
    # 13) Guardar histórico final
    # --------------------------------------------------------
    save_json(history, OUTPUT_DIR / "history.json")

    summary = {
        "best_epoch": best_epoch,
        "best_val_mcc": best_mcc,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
        "model_path": str(OUTPUT_DIR / "best_model.pth"),
        "history_path": str(OUTPUT_DIR / "history.json"),
    }

    save_json(summary, OUTPUT_DIR / "training_summary.json")

    print("\nEntrenamiento finalizado")
    print(f"Épocas entrenadas : {len(history['train_loss'])}")
    print(f"Mejor época       : {best_epoch}")
    print(f"Mejor Val MCC     : {best_mcc:.4f}")
    print(f"Val Loss mejor MCC: {best_val_loss:.4f}")
    print(f"Modelo guardado en: {OUTPUT_DIR / 'best_model.pth'}")
    print(f"Histórico guardado: {OUTPUT_DIR / 'history.json'}")
    print(f"Resumen guardado  : {OUTPUT_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    main()