import os


import pandas as pd

def load_tf66_video_info(excel_path):
    df = pd.read_excel(excel_path)

    # Limpiar nombres de columnas
    df.columns = [c.strip().replace(" ", "") for c in df.columns]

    video_info = {}

    for _, row in df.iterrows():

        # 🔥 SALTAR FILAS BASURA (cabeceras repetidas)
        if str(row["framesBeforeFall"]) == "framesBeforeFall":
            continue

        name = str(row["RecordingName"]).strip().replace(".avi", "")

        frames_before = row.get("framesBeforeFall", None)
        frames_after = row.get("framesAfterFall", None)

        # Convertir solo si son números válidos
        try:
            frames_before = int(frames_before) if pd.notna(frames_before) else None
        except:
            continue

        try:
            frames_after = int(frames_after) if pd.notna(frames_after) else None
        except:
            frames_after = None

        video_info[name] = {
            "framesBeforeFall": frames_before,
            "framesAfterFall": frames_after
        }

    print(f"Vídeos con metadata válida cargados desde Excel: {len(video_info)}")

    return video_info


def build_tf66_samples(root_dir, video_info, seq_len=10):

    samples = []

    for label_name, label_value in [("Fall", 1), ("NonFall", 0)]:
        folder = os.path.join(root_dir, label_name)

        if not os.path.exists(folder):
            continue

        for vid in os.listdir(folder):
            path = os.path.join(folder, vid)
            if not os.path.isdir(path):
                continue

            video_name = vid

            frames = sorted(os.listdir(path))
            total = len(frames)

            if total < seq_len:
                continue

            if label_value == 1 and video_name in video_info:
                info = video_info[video_name]

                if info["framesBeforeFall"] is not None:
                    start = max(0, info["framesBeforeFall"] - seq_len // 2)
                else:
                    start = 0

            else:
                start = max(0, total // 2 - seq_len // 2)

            samples.append({
                "path": path,
                "label": label_value,
                "start": start,
                "seq_len": seq_len
            })

    return samples