import os
import cv2
import random

def get_frames(video_path, class_name):
    # Crear carpetas de salida
    os.makedirs(f'dataset/train/{class_name}', exist_ok=True)
    os.makedirs(f'dataset/validate/{class_name}', exist_ok=True)
    os.makedirs(f'dataset/test/{class_name}', exist_ok=True)

    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No pude abrir el video {video_path}.")
        return

    # Obtener el frame rate del video
    frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
    if frame_rate == 0:
        print("Error: El frame rate del video es 0. El archivo de video podría estar corrupto o mal codificado.")
        cap.release()
        return

    # Definir el intervalo de frames
    frame_interval = max(frame_rate // 1000, 1)

    # Extraer frames del video
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            frames.append(frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

    # Mezclar y dividir los frames en train, validate, y test
    random.shuffle(frames)
    total_frames = len(frames)
    train_split = int(0.7 * total_frames)
    validate_split = int(0.15 * total_frames)

    train_frames = frames[:train_split]
    validate_frames = frames[train_split:train_split + validate_split]
    test_frames = frames[train_split + validate_split:]

    # Guardar frames en las carpetas correspondientes
    def save_frames(frames_list, folder):
        for idx, frame in enumerate(frames_list):
            frame_path = f'{folder}/{class_name}_frame_{idx}.jpg'
            if not cv2.imwrite(frame_path, frame):
                print(f"Error al guardar el frame en {frame_path}")

    save_frames(train_frames, f'dataset/train/{class_name}')
    save_frames(validate_frames, f'dataset/validate/{class_name}')
    save_frames(test_frames, f'dataset/test/{class_name}')


# Extraer frames de los videos y guardarlos en carpetas específicas por clase
get_frames('videos/diego/diego.mp4', 'diego')
get_frames('videos/alex/alex.mp4','alex')
get_frames('videos/inigo/inigo.mp4','inigo')
