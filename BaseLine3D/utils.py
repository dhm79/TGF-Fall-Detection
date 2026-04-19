import os
import re
import pandas as pd


def load_tf66_video_info(excel_path):
    """
    Carga la metadata del Excel original de TF-66.

    IMPORTANTE:
    En tu Excel la cabecera real está en la fila 3,
    por eso usamos header=2.

    Devuelve un diccionario así:
    {
        "01-Fall-01": {
            "framesBeforeFall": 40,
            "framesAfterFall": 70,
            "firstFallFrameOfVideo": 39
        },
        ...
    }
    """
    df = pd.read_excel(excel_path, header=2)

    video_info = {}

    for _, row in df.iterrows():
        video_name = str(row["Recording Name"]).strip()

        before = pd.to_numeric(row["framesBeforeFall"], errors="coerce")
        after = pd.to_numeric(row["framesAfterFall"], errors="coerce")
        first_fall = pd.to_numeric(row["First Fall Frame of Video"], errors="coerce")

        # Saltamos filas vacías o inválidas
        if (
            video_name == ""
            or video_name.lower() == "nan"
            or pd.isna(before)
            or pd.isna(after)
            or pd.isna(first_fall)
        ):
            continue

        video_info[video_name] = {
            "framesBeforeFall": int(before),
            "framesAfterFall": int(after),
            "firstFallFrameOfVideo": int(first_fall),
        }

    print(f"Vídeos con metadata válida cargados desde Excel: {len(video_info)}")
    return video_info


def natural_sort_key(filename):
    """
    Orden natural basada en el último número del nombre.
    """
    numbers = re.findall(r"\d+", filename)
    return int(numbers[-1]) if numbers else 0


def count_valid_frames(path):
    """
    Cuenta cuántos frames válidos hay dentro de una carpeta.
    """
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    frame_files = [
        f for f in os.listdir(path)
        if f.lower().endswith(valid_exts)
    ]
    frame_files = sorted(frame_files, key=natural_sort_key)
    return len(frame_files)


def build_tf66_samples(root_dir, excel_path, seq_len=10, min_frames=None):
    """
    Construye muestras a nivel de vídeo.

    Cada muestra representa una carpeta de frames.
    Después, el Dataset elige la ventana temporal concreta:
    - NonFall: aleatoria
    - Fall: guiada por el Excel

    Parámetros
    ----------
    root_dir:
        Carpeta raíz del split, por ejemplo:
        ../Train
        ../Validation

    excel_path:
        Ruta al 'Final Dataset.xlsx'

    seq_len:
        Número de frames por clip. En el baseline del paper: 10.

    min_frames:
        Mínimo de frames para aceptar una muestra.
        Si no se especifica, usamos seq_len.
    """
    if min_frames is None:
        min_frames = seq_len

    video_info = load_tf66_video_info(excel_path)
    samples = []

    for label_name, label_value in [("Fall", 1), ("NonFall", 0)]:
        folder = os.path.join(root_dir, label_name)

        if not os.path.exists(folder):
            continue

        for item in os.listdir(folder):
            path = os.path.join(folder, item)

            if not os.path.isdir(path):
                continue

            total_frames = count_valid_frames(path)
            if total_frames < min_frames:
                continue

            video_name = item.strip()

            # Para vídeos Fall exigimos metadata en el Excel
            if label_value == 1 and video_name not in video_info:
                continue

            samples.append({
                "path": path,
                "label": label_value,
                "video_name": video_name,
                "total_frames": total_frames,
                "seq_len": seq_len,
            })

    return samples