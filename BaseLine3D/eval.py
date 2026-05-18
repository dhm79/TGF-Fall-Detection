"""
Evaluación del baseline 3D CNN.

1. Usa rutas relativas:
   - ../Validation
   - ../Final Dataset.xlsx
   - ./outputs/best_model.pth

2. Lee automáticamente el Excel original de TF-66.

3. Construye las muestras de validación a partir de las carpetas de frames.

4. Carga el mejor modelo guardado durante el entrenamiento.

5. Evalúa el modelo sobre Validation.

6. Guarda:
   - outputs/eval_metrics.json

Cómo ejecutarlo
---------------
Desde la carpeta BaseLine3D:

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
from model import TF66Baseline3DCNN


# ============================================================
# RUTAS RELATIVAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TF66_ROOT = BASE_DIR.parent

VAL_DIR = TF66_ROOT / "Validation"
EXCEL_PATH = TF66_ROOT / "Final Dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = OUTPUT_DIR / "best_model.pth"


# ============================================================
# HIPERPARÁMETROS
# ============================================================

SEQ_LEN = 10
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 0
SEED = 42
THRESHOLD = 0.5


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def set_seed(seed=42):
    """
    Fija semillas para hacer la evaluación reproducible.
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
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def load_model_weights(model, model_path, device):
    """
    Carga pesos del modelo. Usa weights_only=True si la versión de PyTorch lo soporta.
    """
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    return model


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
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el Excel: {EXCEL_PATH}")

    if not VAL_DIR.exists():
        raise FileNotFoundError(f"No se encontró la carpeta Validation: {VAL_DIR}")

    # --------------------------------------------------------
    # 2) Cargar metadata temporal del Excel
    # --------------------------------------------------------
    video_info = load_tf66_video_info(EXCEL_PATH)

    # --------------------------------------------------------
    # 3) Construir muestras de Validation
    # --------------------------------------------------------
    val_samples = build_tf66_samples(VAL_DIR, EXCEL_PATH, seq_len=SEQ_LEN)

    if len(val_samples) == 0:
        raise ValueError("No se encontraron samples en Validation.")

    print(f"Validation samples: {len(val_samples)}")
    print(f"Modelo cargado desde: {MODEL_PATH}")
    print(f"Seq len  : {SEQ_LEN}")
    print(f"Img size : {IMG_SIZE}")
    print(f"Threshold: {THRESHOLD}")

    # --------------------------------------------------------
    # 4) Crear Dataset y DataLoader
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 5) Crear modelo y cargar pesos
    # --------------------------------------------------------
    model = TF66Baseline3DCNN().to(device)
    model = load_model_weights(model, MODEL_PATH, device)
    model.eval()

    # --------------------------------------------------------
    # 6) Evaluación
    # --------------------------------------------------------
    y_true = []
    y_prob = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

            y_true.extend(y.detach().cpu().numpy().flatten())
            y_prob.extend(probs)

    # --------------------------------------------------------
    # 7) Calcular métricas finales
    # --------------------------------------------------------
    metrics = compute_metrics(y_true, y_prob, threshold=THRESHOLD)

    metrics["threshold"] = THRESHOLD
    metrics["seq_len"] = SEQ_LEN
    metrics["img_size"] = IMG_SIZE
    metrics["batch_size"] = BATCH_SIZE
    metrics["num_validation_samples"] = len(val_samples)
    metrics["model_path"] = str(MODEL_PATH)

    print("\n=== RESULTADOS EN VALIDATION ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"MCC      : {metrics['mcc']:.4f}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    # --------------------------------------------------------
    # 8) Guardar métricas en JSON
    # --------------------------------------------------------
    save_json(metrics, OUTPUT_DIR / "eval_metrics.json")

    print(f"\nMétricas guardadas en: {OUTPUT_DIR / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()