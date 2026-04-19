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
    matthews_corrcoef
)

from dataset import TF66ClipDataset
from utils import build_tf66_samples, load_tf66_video_info
from model import TF66Baseline3DCNN


# ============================================================
# RUTAS RELATIVAS
# ============================================================

# Carpeta donde está este archivo eval.py
BASE_DIR = Path(__file__).resolve().parent

# Carpeta raíz TF66 (la carpeta padre de BaseLine3D)
TF66_ROOT = BASE_DIR.parent

# Subcarpetas / archivos reales del proyecto
VAL_DIR = TF66_ROOT / "Validation"
EXCEL_PATH = TF66_ROOT / "Final Dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"

# Modelo guardado durante train.py
MODEL_PATH = OUTPUT_DIR / "best_model.pth"


# ============================================================
# HIPERPARÁMETROS
# ============================================================

# Deben coincidir con los usados en train.py
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
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Flujo principal de evaluación.
    """
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    # Elegimos dispositivo automáticamente
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # --------------------------------------------------------
    # 1) Comprobar que existen el modelo y el Excel
    # --------------------------------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el Excel: {EXCEL_PATH}")

    # --------------------------------------------------------
    # 2) Cargar metadata temporal del Excel
    # --------------------------------------------------------
    # Esto devuelve información de los vídeos Fall:
    # framesBeforeFall, framesAfterFall, firstFallFrameOfVideo
    video_info = load_tf66_video_info(EXCEL_PATH)

    # --------------------------------------------------------
    # 3) Construir muestras de Validation
    # --------------------------------------------------------
    # build_tf66_samples() recorre:
    #   Validation/Fall
    #   Validation/NonFall
    # y construye una lista de muestras a nivel de vídeo.
    val_samples = build_tf66_samples(VAL_DIR, EXCEL_PATH, seq_len=SEQ_LEN)

    if len(val_samples) == 0:
        raise ValueError("No se encontraron samples en Validation.")

    print(f"Validation samples: {len(val_samples)}")

    # --------------------------------------------------------
    # 4) Crear Dataset y DataLoader
    # --------------------------------------------------------
    # En evaluación usamos random_sampling=False para que la selección
    # del clip sea determinista y no cambie en cada ejecución.
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

    # --------------------------------------------------------
    # 5) Crear modelo y cargar pesos
    # --------------------------------------------------------
    model = TF66Baseline3DCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # --------------------------------------------------------
    # 6) Evaluación
    # --------------------------------------------------------
    y_true = []
    y_prob = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)

            # logits: salida cruda del modelo
            logits = model(x)

            # Convertimos logits a probabilidades con sigmoide
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            y_true.extend(y.numpy().flatten())
            y_prob.extend(probs)

    # --------------------------------------------------------
    # 7) Calcular métricas finales
    # --------------------------------------------------------
    metrics = compute_metrics(y_true, y_prob, threshold=THRESHOLD)

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