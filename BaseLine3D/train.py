"""
Entrenamiento del baseline 3D CNN.

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

5. Entrena el modelo 3D CNN.

6. Guarda:
   - outputs/best_model.pth
   - outputs/history.json

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
import random
from pathlib import Path

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
from model import TF66Baseline3DCNN


# ============================================================
# RUTAS RELATIVAS
# ============================================================

# Carpeta donde está este archivo train.py
BASE_DIR = Path(__file__).resolve().parent

# Carpeta raíz TF66 (la carpeta padre de BaseLine3D)
TF66_ROOT = BASE_DIR.parent

# Subcarpetas reales del proyecto
TRAIN_DIR = TF66_ROOT / "Train"
VAL_DIR = TF66_ROOT / "Validation"
EXCEL_PATH = TF66_ROOT / "Final Dataset.xlsx"

# Carpeta donde se guardarán resultados
OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================
# HIPERPARÁMETROS
# ============================================================

# En el baseline del paper se usan secuencias de 10 frames
SEQ_LEN = 10

# El baseline trabaja con entrada 256x256
IMG_SIZE = 256

# En CPU puede tardar bastante. 
# Se puede probar temporalmente con BATCH_SIZE = 4 o 2.
BATCH_SIZE = 16

# Número de épocas
EPOCHS = 100

# Optimizador Adam con LR del paper
LR = 1e-4

# Sin regularización L2 extra
WEIGHT_DECAY = 0.0

# En Windows, mejor empezar con 0
NUM_WORKERS = 0

# Semilla
SEED = 42

# Umbral de clasificación binaria
THRESHOLD = 0.5


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
    y_pred = [1 if p >= threshold else 0 for p in y_prob]

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

    Devuelve un diccionario con:
    - loss
    - accuracy
    - f1
    - precision
    - recall
    - mcc
    """
    model.train()

    total_loss = 0.0
    y_true = []
    y_prob = []

    for x, y in loader:
        # x: (B, 1, T, H, W)
        # y: (B,)
        x = x.to(device)
        y = y.to(device).unsqueeze(1)  # -> (B, 1)

        optimizer.zero_grad()

        logits = model(x)              # salida cruda, sin sigmoide
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


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Flujo principal del entrenamiento.
    """
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    # Elegimos dispositivo automáticamente
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # --------------------------------------------------------
    # 1) Comprobar que existe el Excel
    # --------------------------------------------------------
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el Excel: {EXCEL_PATH}")

    # --------------------------------------------------------
    # 2) Cargar metadata temporal del Excel
    # --------------------------------------------------------
    # Esto devuelve un diccionario con información de cada vídeo Fall.
    video_info = load_tf66_video_info(EXCEL_PATH)

    # --------------------------------------------------------
    # 3) Construir muestras a nivel de vídeo
    # --------------------------------------------------------
    # OJO:
    # build_tf66_samples() solo devuelve una lista de samples.
    # No devuelve video_info, por eso lo hemos cargado aparte arriba.
    train_samples = build_tf66_samples(TRAIN_DIR, EXCEL_PATH, seq_len=SEQ_LEN)
    val_samples = build_tf66_samples(VAL_DIR, EXCEL_PATH, seq_len=SEQ_LEN)

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    if len(train_samples) == 0:
        raise ValueError("No se encontraron samples de entrenamiento.")

    if len(val_samples) == 0:
        raise ValueError("No se encontraron samples de validación.")

    # Contamos cuántos Fall y NonFall hay en train
    labels = [int(s["label"]) for s in train_samples]
    pos = sum(labels)
    neg = len(labels) - pos

    print(f"Fall train samples: {pos}")
    print(f"NonFall train samples: {neg}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Seq len  : {SEQ_LEN}")
    print(f"Img size : {IMG_SIZE}")
    print(f"Batch    : {BATCH_SIZE}")
    print(f"Epochs   : {EPOCHS}")

    # --------------------------------------------------------
    # 4) Crear Dataset y DataLoader
    # --------------------------------------------------------
    train_ds = TF66ClipDataset(
        train_samples,
        video_info=video_info,
        img_size=IMG_SIZE,
        random_sampling=True,   # aleatorio en train
    )

    val_ds = TF66ClipDataset(
        val_samples,
        video_info=video_info,
        img_size=IMG_SIZE,
        random_sampling=False,  # determinista en validación
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # 5) Crear modelo
    # --------------------------------------------------------
    model = TF66Baseline3DCNN().to(device)

    # Pérdida binaria con logits
    criterion = nn.BCEWithLogitsLoss()

    # Optimizador Adam
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Mejor métrica observada en validación
    best_mcc = float("-inf")

    # Historial para guardar después en JSON
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

    # --------------------------------------------------------
    # 6) Bucle de entrenamiento
    # --------------------------------------------------------
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

        # Guardamos el mejor modelo según MCC en validación
        if val_metrics["mcc"] > best_mcc:
            best_mcc = val_metrics["mcc"]
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
            print("** Mejor modelo guardado por MCC **")

    # --------------------------------------------------------
    # 7) Guardar histórico
    # --------------------------------------------------------
    save_json(history, OUTPUT_DIR / "history.json")

    print("\nEntrenamiento finalizado")
    print(f"Mejor MCC: {best_mcc:.4f}")
    print(f"Modelo guardado en: {OUTPUT_DIR / 'best_model.pth'}")
    print(f"Histórico guardado en: {OUTPUT_DIR / 'history.json'}")


if __name__ == "__main__":
    main()