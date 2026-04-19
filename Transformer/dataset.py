import os
import re
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class TF66ClipDataset(Dataset):
    """
    Dataset de TF-66 con selección de clip basada en el Excel:
    - Fall: ventana válida alrededor de la caída
    - NonFall: ventana aleatoria válida
    """

    def __init__(self, samples, video_info, img_size=128, random_sampling=True):
        self.samples = samples
        self.video_info = video_info
        self.img_size = img_size
        self.random_sampling = random_sampling

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        path = sample["path"]
        label = float(sample["label"])
        video_name = sample["video_name"]
        seq_len = sample["seq_len"]

        frames = self.load_frames(path)
        start = self.choose_start(frames, video_name, seq_len, label)
        clip = self.extract_clip(frames, start, seq_len)
        clip = [self.preprocess(f) for f in clip]

        clip = np.stack(clip, axis=0)  # (T, H, W)
        clip = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)  # (1, T, H, W)

        label = torch.tensor(label, dtype=torch.float32)
        return clip, label

    def _natural_sort_key(self, filename):
        numbers = re.findall(r"\d+", filename)
        return int(numbers[-1]) if numbers else 0

    def load_frames(self, path):
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
        total_frames = len(frames)

        if total_frames < seq_len:
            return 0

        # NonFall
        if label == 0:
            start_min = 0
            start_max = max(0, total_frames - seq_len)

        # Fall
        else:
            if video_name not in self.video_info:
                raise ValueError(f"No hay metadata en Excel para: {video_name}")

            meta = self.video_info[video_name]
            frames_after_fall = meta["framesAfterFall"]
            first_fall_frame = meta["firstFallFrameOfVideo"]

            if seq_len > frames_after_fall:
                raise ValueError(
                    f"seq_len={seq_len} no puede ser mayor que framesAfterFall={frames_after_fall} "
                    f"en {video_name}"
                )

            start_min = max(0, first_fall_frame - seq_len)
            start_max = min(first_fall_frame - (seq_len // 2), total_frames - seq_len)

            if start_min > start_max:
                raise ValueError(
                    f"No hay rango válido para {video_name}: start_min={start_min}, start_max={start_max}"
                )

        if self.random_sampling:
            return random.randint(start_min, start_max)

        return (start_min + start_max) // 2

    def extract_clip(self, frames, start, seq_len):
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
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = frame.astype(np.float32) / 255.0
        return frame