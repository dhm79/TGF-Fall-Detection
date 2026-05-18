"""
Prueba real / simulación en tiempo real para modelos TF-66.

Permite usar:
1. Baseline 3D CNN:
       python real_test.py --model_type baseline3d --source video --video_path ruta/video.avi

2. Transformer:
       python real_test.py --model_type transformer --source video --video_path ruta/video.avi

3. Cámara USB:
       python real_test.py --model_type baseline3d --source camera --camera_index 0

Entrada esperada por ambos modelos:
    (B, 1, T, H, W)

Sistema de alertas:
    prob < LOW_THRESHOLD:
        No caída

    LOW_THRESHOLD <= prob < HIGH_THRESHOLD:
        Alerta preventiva / posible caída

    prob >= HIGH_THRESHOLD:
        Alerta urgente inmediata

    Si la posible caída se mantiene durante CONFIRM_SECONDS:
        Alerta urgente confirmada
"""

import argparse
import importlib
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import torch


# ============================================================
# CONFIGURACIÓN BASE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_MODEL_PATH = OUTPUT_DIR / "best_model.pth"

SEQ_LEN = 10
IMG_SIZE = 256
TARGET_FPS = 4


# ============================================================
# CONFIGURACIÓN DE ALERTAS
# ============================================================

LOW_THRESHOLD = 0.50
HIGH_THRESHOLD = 0.85
CONFIRM_SECONDS = 10.0
ALERT_COOLDOWN = 60.0

TELEGRAM_ENABLED = False
TELEGRAM_BOT_TOKEN = "PON_AQUI_TU_TOKEN"
TELEGRAM_CHAT_ID = "PON_AQUI_TU_CHAT_ID"


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram_alert(message):
    if not TELEGRAM_ENABLED:
        return

    if TELEGRAM_BOT_TOKEN == "PON_AQUI_TU_TOKEN":
        print("[AVISO] Telegram no configurado: falta TELEGRAM_BOT_TOKEN")
        return

    if TELEGRAM_CHAT_ID == "PON_AQUI_TU_CHAT_ID":
        print("[AVISO] Telegram no configurado: falta TELEGRAM_CHAT_ID")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(url, data=payload, timeout=5)

        if response.status_code != 200:
            print(f"[ERROR TELEGRAM] {response.text}")

    except Exception as e:
        print(f"[ERROR TELEGRAM] No se pudo enviar alerta: {e}")


# ============================================================
# PREPROCESADO
# ============================================================

def preprocess_frame(frame, img_size=256):
    if frame is None:
        raise ValueError("Frame vacío recibido.")

    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame.astype(np.float32) / 255.0

    return frame


def make_clip_tensor(frames, device):
    """
    Salida:
        (B, C, T, H, W) = (1, 1, T, H, W)
    """

    clip = np.stack(frames, axis=0)          # (T, H, W)
    clip = torch.tensor(clip, dtype=torch.float32)

    clip = clip.unsqueeze(0)                 # (1, T, H, W)
    clip = clip.unsqueeze(0)                 # (B, C, T, H, W)

    return clip.to(device)


# ============================================================
# MODELOS
# ============================================================

def load_model(
    model_path,
    device,
    model_type,
    img_size,
    seq_len,
    patch_size,
    embed_dim,
    spatial_depth,
    temporal_depth,
    num_heads,
    mlp_ratio,
    dropout,
):
    """
    Carga uno de estos modelos desde model.py:

    - baseline3d:
        necesita una clase llamada TF66Baseline3DCNN

    - transformer:
        necesita una clase llamada FallTransformer
    """

    model_module = importlib.import_module("model")

    if model_type == "baseline3d":
        if not hasattr(model_module, "TF66Baseline3DCNN"):
            raise ImportError(
                "No se encontró la clase TF66Baseline3DCNN en model.py. "
                "Ejecuta este script desde la carpeta BaseLine3D o usa --model_type transformer."
            )

        model_class = getattr(model_module, "TF66Baseline3DCNN")
        model = model_class().to(device)

    elif model_type == "transformer":
        if not hasattr(model_module, "FallTransformer"):
            raise ImportError(
                "No se encontró la clase FallTransformer en model.py. "
                "Ejecuta este script desde la carpeta Transformer o usa --model_type baseline3d."
            )

        model_class = getattr(model_module, "FallTransformer")

        model = model_class(
            img_size=img_size,
            patch_size=patch_size,
            seq_len=seq_len,
            in_chans=1,
            embed_dim=embed_dim,
            spatial_depth=spatial_depth,
            temporal_depth=temporal_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        ).to(device)

    else:
        raise ValueError(f"model_type no soportado: {model_type}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.no_grad()
def predict_clip(model, frames, device):
    clip_tensor = make_clip_tensor(frames, device)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    logits = model(clip_tensor)
    prob = torch.sigmoid(logits).item()

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    inference_ms = (end - start) * 1000.0

    return prob, inference_ms


# ============================================================
# ALERTAS
# ============================================================

def classify_alert_level(prob, low_threshold, high_threshold):
    if prob >= high_threshold:
        return "urgent"

    if prob >= low_threshold:
        return "preventive"

    return "none"


def handle_alert_logic(
    prob,
    inference_ms,
    low_threshold,
    high_threshold,
    confirm_seconds,
    alert_cooldown,
    state,
):
    now = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    level = classify_alert_level(prob, low_threshold, high_threshold)

    if level == "urgent":
        label_text = "CAIDA URGENTE"
        is_fall = True
        alert_level = "urgent"

        print(
            f"[ALERTA URGENTE] prob={prob:.3f} | "
            f"inferencia={inference_ms:.2f} ms"
        )

        if now - state["last_urgent_alert_time"] >= alert_cooldown:
            send_telegram_alert(
                f"ALERTA URGENTE: posible caída detectada\n"
                f"Hora: {timestamp}\n"
                f"Probabilidad: {prob:.3f}\n"
                f"Tipo: caída segura"
            )
            state["last_urgent_alert_time"] = now

        state["possible_fall_active"] = False
        state["possible_fall_start_time"] = None

        return label_text, is_fall, alert_level

    if level == "preventive":
        label_text = "POSIBLE CAIDA"
        is_fall = True
        alert_level = "preventive"

        print(
            f"[PREVENTIVA] prob={prob:.3f} | "
            f"inferencia={inference_ms:.2f} ms"
        )

        if not state["possible_fall_active"]:
            state["possible_fall_active"] = True
            state["possible_fall_start_time"] = now

            if now - state["last_preventive_alert_time"] >= alert_cooldown:
                send_telegram_alert(
                    f"ALERTA PREVENTIVA: posible caída\n"
                    f"Hora: {timestamp}\n"
                    f"Probabilidad: {prob:.3f}\n"
                    f"Estado: esperando confirmación durante {confirm_seconds:.0f} s"
                )
                state["last_preventive_alert_time"] = now

            return label_text, is_fall, alert_level

        elapsed_possible = now - state["possible_fall_start_time"]

        if elapsed_possible >= confirm_seconds:
            label_text = "CAIDA CONFIRMADA"
            alert_level = "confirmed"

            print(
                f"[ALERTA URGENTE CONFIRMADA] "
                f"posible caída mantenida {elapsed_possible:.1f} s | "
                f"prob={prob:.3f}"
            )

            if now - state["last_urgent_alert_time"] >= alert_cooldown:
                send_telegram_alert(
                    f"ALERTA URGENTE CONFIRMADA\n"
                    f"Hora: {timestamp}\n"
                    f"La persona podría seguir inmóvil o en el suelo.\n"
                    f"Tiempo en posible caída: {elapsed_possible:.1f} s\n"
                    f"Probabilidad actual: {prob:.3f}"
                )
                state["last_urgent_alert_time"] = now

        return label_text, is_fall, alert_level

    label_text = "No caida"
    is_fall = False
    alert_level = "none"

    print(
        f"No caida | prob={prob:.3f} | "
        f"inferencia={inference_ms:.2f} ms"
    )

    state["possible_fall_active"] = False
    state["possible_fall_start_time"] = None

    return label_text, is_fall, alert_level


# ============================================================
# VISUALIZACIÓN
# ============================================================

def draw_info(
    display,
    label_text,
    prob,
    inference_ms,
    low_threshold,
    high_threshold,
    confirm_seconds,
    model_type,
):
    if label_text in ["CAIDA URGENTE", "CAIDA CONFIRMADA"]:
        color = (0, 0, 255)
    elif label_text == "POSIBLE CAIDA":
        color = (0, 165, 255)
    else:
        color = (0, 255, 0)

    cv2.putText(
        display,
        label_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        display,
        f"Modelo: {model_type}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Probabilidad caida: {prob:.3f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Inferencia: {inference_ms:.2f} ms",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Low: {low_threshold:.2f} | High: {high_threshold:.2f}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Confirmacion: {confirm_seconds:.0f} s",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return display


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

def run_realtime(
    source,
    video_path,
    camera_index,
    model_path,
    model_type,
    target_fps,
    img_size,
    seq_len,
    low_threshold,
    high_threshold,
    confirm_seconds,
    alert_cooldown,
    no_display,
    patch_size,
    embed_dim,
    spatial_depth,
    temporal_depth,
    num_heads,
    mlp_ratio,
    dropout,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    model = load_model(
        model_path=model_path,
        device=device,
        model_type=model_type,
        img_size=img_size,
        seq_len=seq_len,
        patch_size=patch_size,
        embed_dim=embed_dim,
        spatial_depth=spatial_depth,
        temporal_depth=temporal_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )

    print(f"Modelo cargado desde: {model_path}")
    print(f"Tipo de modelo: {model_type}")

    if source == "video":
        if video_path is None:
            raise ValueError("Debes pasar --video_path si source=video")

        cap = cv2.VideoCapture(str(video_path))
        print(f"Abriendo vídeo: {video_path}")

    elif source == "camera":
        cap = cv2.VideoCapture(camera_index)
        print(f"Abriendo cámara index: {camera_index}")

    else:
        raise ValueError("source debe ser 'video' o 'camera'")

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la fuente de vídeo/cámara.")

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps is None or source_fps <= 0 or source == "camera":
        source_fps = target_fps

    frame_step = max(1, round(source_fps / target_fps))

    print(f"FPS fuente aproximado: {source_fps:.2f}")
    print(f"FPS objetivo: {target_fps}")
    print(f"Usando 1 de cada {frame_step} frames")
    print(f"Seq len: {seq_len}")
    print(f"Img size: {img_size}")
    print(f"Low threshold: {low_threshold}")
    print(f"High threshold: {high_threshold}")
    print(f"Confirm seconds: {confirm_seconds}")
    print(f"Alert cooldown: {alert_cooldown}")
    print(f"No display: {no_display}")

    if model_type == "transformer":
        print("Configuración Transformer:")
        print(f"  patch_size: {patch_size}")
        print(f"  embed_dim: {embed_dim}")
        print(f"  spatial_depth: {spatial_depth}")
        print(f"  temporal_depth: {temporal_depth}")
        print(f"  num_heads: {num_heads}")
        print(f"  mlp_ratio: {mlp_ratio}")
        print(f"  dropout: {dropout}")

    frame_buffer = deque(maxlen=seq_len)

    frame_idx = 0
    processed_frames = 0

    last_prob = 0.0
    last_inference_ms = 0.0
    last_label = "Cargando buffer..."
    last_is_fall = False

    alert_state = {
        "possible_fall_active": False,
        "possible_fall_start_time": None,
        "last_preventive_alert_time": 0.0,
        "last_urgent_alert_time": 0.0,
    }

    start_global = time.perf_counter()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Fin de la fuente de vídeo.")
            break

        frame_idx += 1

        if frame_idx % frame_step != 0:
            continue

        processed_frames += 1

        processed = preprocess_frame(frame, img_size=img_size)
        frame_buffer.append(processed)

        display = (processed * 255).astype(np.uint8)
        display = cv2.resize(display, (500, 500))
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        if len(frame_buffer) < seq_len:
            last_label = f"Cargando buffer: {len(frame_buffer)}/{seq_len}"
            last_prob = 0.0
            last_inference_ms = 0.0
            last_is_fall = False

        else:
            prob, inference_ms = predict_clip(
                model=model,
                frames=list(frame_buffer),
                device=device
            )

            last_prob = prob
            last_inference_ms = inference_ms

            last_label, last_is_fall, _ = handle_alert_logic(
                prob=prob,
                inference_ms=inference_ms,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                confirm_seconds=confirm_seconds,
                alert_cooldown=alert_cooldown,
                state=alert_state,
            )

        display = draw_info(
            display=display,
            label_text=last_label,
            prob=last_prob,
            inference_ms=last_inference_ms,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            confirm_seconds=confirm_seconds,
            model_type=model_type,
        )

        if not no_display:
            cv2.imshow("Prueba real - deteccion de caidas", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Salida manual.")
                break

    end_global = time.perf_counter()
    elapsed = end_global - start_global

    cap.release()

    if not no_display:
        cv2.destroyAllWindows()

    if elapsed > 0:
        effective_fps = processed_frames / elapsed
        print("\n=== RESUMEN ===")
        print(f"Frames procesados: {processed_frames}")
        print(f"Tiempo total: {elapsed:.2f} s")
        print(f"FPS efectivo aproximado: {effective_fps:.2f}")


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline3d",
        choices=["baseline3d", "transformer"],
        help="Tipo de modelo a cargar"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="video",
        choices=["video", "camera"],
        help="Fuente de entrada: video o camera"
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Ruta al vídeo térmico .avi/.mp4"
    )

    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Índice de cámara para OpenCV"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Ruta al modelo .pth. Si no se indica, usa outputs/best_model.pth"
    )

    parser.add_argument(
        "--target_fps",
        type=float,
        default=TARGET_FPS,
        help="FPS objetivo para aproximar TF-66"
    )

    # Los dejamos en None para rellenarlos según el modelo.
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)

    parser.add_argument("--low_threshold", type=float, default=None)
    parser.add_argument("--high_threshold", type=float, default=None)

    parser.add_argument(
        "--confirm_seconds",
        type=float,
        default=CONFIRM_SECONDS,
        help="Segundos que debe mantenerse una posible caída para confirmarla"
    )

    parser.add_argument(
        "--alert_cooldown",
        type=float,
        default=ALERT_COOLDOWN,
        help="Segundos mínimos entre alertas para evitar spam"
    )

    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Desactiva cv2.imshow. Útil para Colab o servidores sin pantalla."
    )

    # Parámetros Transformer. También se rellenan automáticamente.
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--spatial_depth", type=int, default=None)
    parser.add_argument("--temporal_depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--mlp_ratio", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    args = parser.parse_args()

    # --------------------------------------------------------
    # Perfiles por defecto según modelo
    # --------------------------------------------------------
    MODEL_DEFAULTS = {
        "baseline3d": {
            "seq_len": 10,
            "img_size": 256,
            "low_threshold": 0.50,
            "high_threshold": 0.85,

            # No se usan en baseline, pero ponemos valores válidos.
            "patch_size": 16,
            "embed_dim": 128,
            "spatial_depth": 2,
            "temporal_depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
        },

        "transformer": {
            "seq_len": 10,
            "img_size": 256,
            "low_threshold": 0.50,
            "high_threshold": 0.95,

            # Valores reales de tu checkpoint Transformer.
            "patch_size": 16,
            "embed_dim": 96,
            "spatial_depth": 1,
            "temporal_depth": 1,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.2,
        },
    }

    defaults = MODEL_DEFAULTS[args.model_type]

    # Si el usuario no pasa un valor, usamos el del perfil.
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # Modelo por defecto: outputs/best_model.pth en la carpeta actual.
    if args.model_path is None:
        args.model_path = str(DEFAULT_MODEL_PATH)

    return args

if __name__ == "__main__":
    args = parse_args()

    run_realtime(
        source=args.source,
        video_path=args.video_path,
        camera_index=args.camera_index,
        model_path=args.model_path,
        model_type=args.model_type,
        target_fps=args.target_fps,
        img_size=args.img_size,
        seq_len=args.seq_len,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        confirm_seconds=args.confirm_seconds,
        alert_cooldown=args.alert_cooldown,
        no_display=args.no_display,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )