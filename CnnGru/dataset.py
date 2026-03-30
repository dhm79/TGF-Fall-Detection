import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# Esta clase hereda de Dataset de PyTorch
# Sirve para definir cómo se cargan y procesan los datos
class TF66ClipDataset(Dataset):

    def __init__(self, clips, img_size=128):
        # clips: lista generada por build_tf66_clips (cada elemento es un clip)
        # img_size: tamaño al que se redimensionan las imágenes
        self.clips = clips
        self.img_size = img_size

    def __len__(self):
        # Devuelve el número total de clips disponibles
        return len(self.clips)

    def __getitem__(self, idx):
        # Este método se ejecuta cada vez que el DataLoader pide una muestra

        # Obtenemos la información del clip
        clip_info = self.clips[idx]

        # Ruta a la carpeta que contiene los frames
        path = clip_info["path"]

        # Etiqueta (0 = NonFall, 1 = Fall)
        label = float(clip_info["label"])

        # Índice de inicio del clip dentro del vídeo
        start = clip_info["start"]

        # Número de frames que queremos coger
        seq_len = clip_info["seq_len"]

        # -------------------------
        # 1. Cargar todos los frames del vídeo
        # -------------------------
        frames = self.load_frames(path)

        # -------------------------
        # 2. Extraer solo el clip (subsecuencia)
        # -------------------------
        frames = self.extract_clip(frames, start, seq_len)

        # -------------------------
        # 3. Preprocesar cada frame
        # -------------------------
        frames = [self.preprocess(f) for f in frames]

        # -------------------------
        # 4. Convertir a tensor
        # -------------------------
        # np.stack → (T, H, W)
        frames = np.stack(frames, axis=0)

        # Convertimos a tensor PyTorch y añadimos canal → (1, T, H, W)
        frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)

        # Convertimos la etiqueta a tensor
        label = torch.tensor(label, dtype=torch.float32)

        # Devolvemos entrada y etiqueta
        return frames, label


    # -------------------------
    # ORDEN NATURAL DE ARCHIVOS
    # -------------------------
    def _natural_sort_key(self, filename):
        # Extrae números del nombre del archivo para ordenar correctamente
        # Ejemplo: frame2.png antes que frame10.png
        numbers = re.findall(r"\d+", filename)
        return int(numbers[-1]) if numbers else 0


    # -------------------------
    # CARGA DE FRAMES
    # -------------------------
    def load_frames(self, path):

        # Comprobamos que la ruta es una carpeta
        if not os.path.isdir(path):
            raise ValueError(f"Se esperaba una carpeta de frames, no: {path}")

        # Filtramos solo archivos de imagen
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
        file_names = [
            f for f in os.listdir(path)
            if f.lower().endswith(valid_exts)
        ]

        # Ordenamos correctamente los frames (muy importante)
        file_names = sorted(file_names, key=self._natural_sort_key)

        # Construimos rutas completas
        files = [os.path.join(path, f) for f in file_names]

        frames = []

        # Leemos cada imagen en escala de grises
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

            # Solo añadimos si se ha leído correctamente
            if img is not None:
                frames.append(img)

        # Si no se ha podido leer nada → error
        if len(frames) == 0:
            raise ValueError(f"No se pudieron leer imágenes en: {path}")

        return frames


    # -------------------------
    # EXTRACCIÓN DEL CLIP
    # -------------------------
    def extract_clip(self, frames, start, seq_len):

        total = len(frames)

        if total == 0:
            raise ValueError("No se han podido leer frames.")

        # Calculamos el final del clip
        end = start + seq_len

        # Si el clip se sale del vídeo
        if end > total:
            # Cogemos lo que queda
            clip = frames[start:]

            # Rellenamos repitiendo el último frame
            # Esto asegura que todos los clips tengan el mismo tamaño
            clip = clip + [clip[-1]] * (seq_len - len(clip))

            return clip

        # Caso normal: devolvemos la subsecuencia
        return frames[start:end]


    # -------------------------
    # PREPROCESADO
    # -------------------------
    def preprocess(self, frame):

        # Redimensionamos la imagen (por ejemplo a 96x96)
        frame = cv2.resize(frame, (self.img_size, self.img_size))

        # Convertimos a float y normalizamos [0,1]
        frame = frame.astype(np.float32) / 255.0

        return frame