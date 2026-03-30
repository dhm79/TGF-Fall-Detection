import os

def build_tf66_clips(root_dir, seq_len=10, stride=10, min_frames=None):
    """
    Genera clips a partir de cada carpeta de frames.

    Parámetros:
    - root_dir: carpeta raíz del conjunto de datos, por ejemplo "./Train" o "./Validation"
    - seq_len: número de frames que tendrá cada clip
    - stride: salto entre un clip y el siguiente
    - min_frames: número mínimo de frames que debe tener una carpeta para usarse;
                  si no se indica, se usa seq_len

    Devuelve:
    - clips: lista de diccionarios, donde cada diccionario representa un clip
    """

    # Si el usuario no ha indicado min_frames,
    # exigimos como mínimo seq_len frames para poder formar un clip completo
    if min_frames is None:
        min_frames = seq_len

    # Aquí iremos guardando todos los clips generados
    clips = []

    # Recorremos las dos clases del problema:
    # "Fall" tendrá etiqueta 1
    # "NonFall" tendrá etiqueta 0
    for label_name, label_value in [("Fall", 1), ("NonFall", 0)]:

        # Construimos la ruta a la carpeta de la clase
        # Ejemplo: "./Train/Fall" o "./Train/NonFall"
        folder = os.path.join(root_dir, label_name)

        # Si esa carpeta no existe, la saltamos
        if not os.path.exists(folder):
            continue

        # Recorremos todo lo que haya dentro de esa carpeta
        # Cada "item" debería ser una carpeta que contiene los frames de un vídeo
        for item in os.listdir(folder):
            path = os.path.join(folder, item)

            # Si no es una carpeta, la ignoramos
            # Esto evita procesar archivos sueltos por error
            if not os.path.isdir(path):
                continue

            # Obtenemos los archivos de imagen dentro de la carpeta
            # Solo aceptamos extensiones típicas de imágenes
            frame_files = [
                f for f in os.listdir(path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]

            # Contamos cuántos frames hay en esa carpeta
            total_frames = len(frame_files)

            # Si la carpeta no tiene suficientes frames, la ignoramos
            # Por ejemplo, si seq_len=10 y una carpeta solo tiene 6 imágenes
            if total_frames < min_frames:
                continue

            # -----------------------------------------
            # Generar varios clips por vídeo
            # -----------------------------------------
            # Vamos recorriendo el vídeo en saltos de "stride"
            # Ejemplo:
            # si total_frames = 120, seq_len = 10 y stride = 10
            # generaremos clips empezando en:
            # 0, 10, 20, 30, ..., 110
            #
            # max(1, total_frames - seq_len + 1) asegura que el rango no quede vacío
            for start in range(0, max(1, total_frames - seq_len + 1), stride):

                # Añadimos un diccionario que describe un clip
                clips.append({
                    "path": path,           # carpeta donde están los frames
                    "label": label_value,   # 1 si es Fall, 0 si es NonFall
                    "start": start,         # índice del frame inicial del clip
                    "seq_len": seq_len      # longitud del clip
                })

            # -----------------------------------------
            # Asegurar el último clip
            # -----------------------------------------
            # Esto sirve para no perder la parte final del vídeo
            #
            # Por ejemplo:
            # total_frames = 117, seq_len = 10, stride = 10
            # los starts serían: 0, 10, 20, ..., 100
            # pero aún quedaría una parte final hasta 107
            #
            # Entonces calculamos el último inicio posible
            last_start = total_frames - seq_len

            # Si ese valor es válido, construimos el último clip
            if last_start >= 0:
                last_clip = {
                    "path": path,
                    "label": label_value,
                    "start": last_start,
                    "seq_len": seq_len
                }

                # Lo añadimos solo si no coincide con el último ya generado
                # Esto evita duplicar el último clip cuando cae exactamente en el stride
                if len(clips) == 0 or clips[-1] != last_clip:
                    clips.append(last_clip)

    # Al final devolvemos la lista completa de clips generados
    return clips