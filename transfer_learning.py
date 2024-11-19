import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import cv2
from tensorflow.keras import layers, models


class_names = ['Alex', 'Diego', 'Inigo']

model = models.load_model('model9.h5')


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error: No se puede abrir la cámara')
    exit()
    
while True:
    ret, frame = cap.read()
    
    if not ret:
        print('Error: No se puede recibir el frame')
        break
    
    size = (224, 224)
    img = cv2.resize(frame, size)
    
    img = img.astype('float32') / 255.0
    img_batch = np.expand_dims(img, axis=0)
    
    results = model(img_batch)
    # print(results)
    
    predicted_index = np.argmax(results)
    predicted_label = class_names[predicted_index]
    # print(predicted_index)
    
    height, width, _ = frame.shape
    
    # Dibujar barras de progreso
    # Configuración de estilo
    bar_height = 20
    bar_color = (0, 255, 0)         # Color del relleno de la barra (verde)
    text_color = (255, 255, 255)     # Color del texto (blanco)
    border_color = (0, 0, 0)         # Color del borde de la barra (negro)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Generación de la barra de resultados
    for i, class_name in enumerate(class_names):
        # Calcula el ancho de la barra basado en la confianza
        bar_width = int(results[0][i] * width)

        # Dibuja la barra de confianza con borde
        top_left = (0, i * bar_height)
        bottom_right = (bar_width, (i + 1) * bar_height)
        cv2.rectangle(frame, top_left, bottom_right, bar_color, -1)  # Relleno de barra
        cv2.rectangle(frame, top_left, bottom_right, border_color, 1)  # Borde de barra

        # Añade el texto con la clase y la confianza
        text_position = (10, (i + 1) * bar_height - 5)
        label = f'{class_name}: {results[0][i]:.2f}'
        cv2.putText(frame, label, text_position, font, font_scale, text_color, font_thickness)
    
    
    text = f'{predicted_label}'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height - 20  # 20 pixels from the bottom
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    cv2.imshow('Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cap.destroyAllWindows()