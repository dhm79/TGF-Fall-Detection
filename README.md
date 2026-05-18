# Fall Detection using Deep Learning (TFG)

Este proyecto implementa un sistema de **detección de caídas** a partir de secuencias de imágenes térmicas utilizando técnicas de **Deep Learning**.

El enfoque combina modelos espaciales y temporales para analizar vídeos completos mediante la segmentación en clips.

---

## Dataset

El dataset **no está incluido** en este repositorio.

Se espera la siguiente estructura:

```
Train/
│
├── Fall/
│   ├── video_1/
│   ├── video_2/
│   └── ...
│
├── NonFall/
│   ├── video_1/
│   └── ...
```

Cada vídeo está representado como una carpeta con múltiples frames (imágenes).

---

## Metodología

El sistema sigue los siguientes pasos:

1. **Segmentación en clips**

   * Cada vídeo se divide en secuencias de longitud fija (`seq_len`)
   * Se utiliza un `stride` para generar múltiples clips por vídeo

2. **Extracción de características (CNN)**

   * Se procesan los frames individualmente
   * Se obtiene un vector de características por frame

3. **Modelado temporal**

   * CNN + GRU (baseline)
   * CNN + Transformer (modelo avanzado)

4. **Clasificación**

   * Predicción binaria: caída / no caída

5. **Agregación por vídeo**

   * Se combinan las predicciones de los clips
   * Se usa la media de probabilidades

---

## Modelos implementados

### BASELINE 3D CNN

* Modelo base
* Captura dependencias temporales secuenciales
* Más ligero y rápido

### Transformer

* Modelo avanzado
* Captura relaciones globales entre frames
* Mejor rendimiento en hardware potente

---

## Entrenamiento

Ejecutar:

```
cd modelo
python train.py
```

Parámetros configurables:

* `SEQ_LEN`: número de frames por clip
* `STRIDE`: separación entre clips
* `BATCH_SIZE`
* `EPOCHS`
* `THRESHOLD`

---

## Evaluación

```
python evaluate.py
```

Se calculan:

* Accuracy
* F1-score
* Precision
* Recall
* MCC
* Matriz de confusión

Resultados guardados en:

```
outputs/eval_metrics.json
```

---

## Notas importantes

* El entrenamiento se realiza sobre clips, no sobre vídeos completos
* La evaluación final se puede realizar:

  * por clip
  * por vídeo (agregación de clips)

---
