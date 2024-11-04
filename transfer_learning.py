import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from tensorflow.keras.applications import MobileNetV2
import cv2
import os
import random
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class_names = ['Diego', 'Alex', 'Inigo']

model = models.load_model('model.h5')


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
    
    # img = img.astype('float32') / 255.0
    img_batch = np.expand_dims(img, axis=0)
    
    results = model(img_batch)
    print(results)
    
    predicted_index = np.argmax(results)
    predicted_label = class_names[predicted_index]
    print(predicted_index)
    
    height, width, _ = frame.shape
    # text = f'{class_name[0]} {class_name[0]:.2f}'
    text = f'{predicted_label}'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height - 20  # 20 pixels from the bottom
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cap.destroyAllWindows()