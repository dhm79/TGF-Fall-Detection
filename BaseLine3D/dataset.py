import os
import re
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TF66ClipDataset(Dataset):
    """
    Dataset de TF-66 usando el Excel original para seleccionar clips.

    Idea general:
    - Fall:
        usamos la info del Excel para elegir una ventana válida que incluya la caída y su contexto.
    - NonFall:
        elegimos una ventana válida cualquiera dentro del vídeo.

    Salida de cada muestra:
    - clip: tensor con forma (1, T, H, W)
    - label: tensor escalar (0.0 o 1.0)
    """

    def __init__(self, samples, video_info, img_size=256, random_sampling=True):
        """
        Parámetros
        ----------
        samples:
            Lista de vídeos construida en utils.py.
        video_info:
            Diccionario con la metadata cargada desde el Excel.
        img_size:
            Tamaño final de cada frame. Para el baseline 3D del paper,
            lo correcto es 256.
        random_sampling:
            - True  -> muestreo aleatorio (para entrenamiento)
            - False -> muestreo determinista/centrado (para evaluación)
        """
        self.samples = samples
        self.video_info = video_info
        self.img_size = img_size
        self.random_sampling = random_sampling

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Devuelve una muestra lista para el modelo.

        Flujo:
        1. Recupera información del vídeo
        2. Carga todos los frames de la carpeta
        3. Elige un punto de inicio válido
        4. Extrae un clip de seq_len frames
        5. Redimensiona y normaliza
        6. Devuelve tensor (1, T, H, W)
        """
        sample = self.samples[idx]

        path = sample["path"]
        label = float(sample["label"])
        video_name = sample["video_name"]
        seq_len = sample["seq_len"]

        frames = self.load_frames(path)
        start = self.choose_start(frames, video_name, seq_len, label)
        clip = self.extract_clip(frames, start, seq_len)
        clip = [self.preprocess(f) for f in clip]

        # clip -> (T, H, W)
        clip = np.stack(clip, axis=0)

        # Añadimos canal -> (1, T, H, W)
        clip = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.float32)
        return clip, label

    def _natural_sort_key(self, filename):
        """
        Orden natural para nombres tipo frame1, frame2, frame10...
        """
        numbers = re.findall(r"\d+", filename)
        return int(numbers[-1]) if numbers else 0

    def load_frames(self, path):
        """
        Lee todos los frames de una carpeta de imágenes.
        """
        if not os.path.isdir(path):
            raise ValueError(f"Se esperaba una carpeta de frames, no: {path}")

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
        file_names = [
            f for f in os.listdir(path)
            if f.lower().endswith(valid_exts)
        ]

        file_names = sorted(file_names, key=self._natural_sort_key)
        files = [os.path.join(path, f) for f in file_names]

        frames = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                frames.append(img)

        if len(frames) == 0:
            raise ValueError(f"No se pudieron leer imágenes en: {path}")

        return frames

    def choose_start(self, frames, video_name, seq_len, label):
        """
        Elige el frame inicial del clip.

        CASO NONFALL
        ------------
        Se elige cualquier ventana válida dentro del vídeo.

        CASO FALL
        ---------
        Se usa la info del Excel:
        - framesBeforeFall
        - framesAfterFall
        - firstFallFrameOfVideo

        La idea es:
        1. construir la región temporal relevante del vídeo
        2. obligar a que el clip contenga el primer frame de caída
        """
        total_frames = len(frames)

        if total_frames < seq_len:
            return 0

        # ----------------------------------------------------
        # NONFALL
        # ----------------------------------------------------
        if label == 0:
            start_min = 0
            start_max = max(0, total_frames - seq_len)

        # ----------------------------------------------------
        # FALL
        # ----------------------------------------------------
        else:
            if video_name not in self.video_info:
                raise ValueError(f"No hay metadata en Excel para: {video_name}")

            meta = self.video_info[video_name]

            frames_before_fall = meta["framesBeforeFall"]
            frames_after_fall = meta["framesAfterFall"]
            first_fall_frame = meta["firstFallFrameOfVideo"]

            # Región relevante marcada por el Excel
            # Empieza antes de la caída y termina después.
            selection_start = max(0, first_fall_frame - frames_before_fall)
            selection_end_exclusive = min(total_frames, first_fall_frame + frames_after_fall)

            # Para que el clip contenga el frame donde empieza la caída:
            # start <= first_fall_frame <= start + seq_len - 1
            fall_constraint_min = first_fall_frame - seq_len + 1
            fall_constraint_max = first_fall_frame

            # Para que el clip quede dentro de la región seleccionada por el Excel
            excel_constraint_min = selection_start
            excel_constraint_max = selection_end_exclusive - seq_len

            # Y también dentro del vídeo real
            video_constraint_min = 0
            video_constraint_max = total_frames - seq_len

            start_min = max(video_constraint_min, excel_constraint_min, fall_constraint_min)
            start_max = min(video_constraint_max, excel_constraint_max, fall_constraint_max)

            if start_min > start_max:
                raise ValueError(
                    f"No hay rango válido para {video_name}: "
                    f"start_min={start_min}, start_max={start_max}, "
                    f"selection_start={selection_start}, "
                    f"selection_end_exclusive={selection_end_exclusive}, "
                    f"first_fall_frame={first_fall_frame}, total_frames={total_frames}"
                )

        if self.random_sampling:
            return random.randint(start_min, start_max)

        # En evaluación usamos el punto medio para que sea determinista
        return (start_min + start_max) // 2

    def extract_clip(self, frames, start, seq_len):
        """
        Extrae un clip de longitud fija.

        Si el clip se sale al final, repite el último frame.
        """
        total = len(frames)

        if total == 0:
            raise ValueError("No se han podido leer frames.")

        end = start + seq_len

        if end > total:
            clip = frames[start:]
            clip = clip + [clip[-1]] * (seq_len - len(clip))
            return clip

        return frames[start:end]

    def preprocess(self, frame):
        """
        Preprocesado sencillo:
        - resize a img_size x img_size
        - normalización a [0, 1]
        """
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = frame.astype(np.float32) / 255.0
        return frame