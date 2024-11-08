import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, mean_squared_error, classification_report

import numpy as np
import matplotlib.pyplot as plt


def prepare_data(train_dir, validate_dir, test_dir, img_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, brightness_range=(0.8, 1.2), horizontal_flip=True)
    validate_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=img_size,
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

    validate_generator = validate_datagen.flow_from_directory(validate_dir,
                                                              target_size=img_size,
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=img_size,
                                                      batch_size=batch_size,
                                                      class_mode='categorical')

    return train_generator, validate_generator, test_generator


# Preprocesamiento de imágenes
train_gen, val_gen, test_gen = prepare_data('./dataset/train',
                                                './dataset/validate',
                                                './dataset/test')


# # Visualizar un lote de imágenes
# plt.figure(figsize=(10, 10))
# plt.imshow(train_gen[0][0][0])
# plt.show()

# Modelo
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False


model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')  # 2 clases: persona/no persona
])

print(model.summary())


model.compile(optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])


early_sttoping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

 # Entrenamiento del modelo
model_history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=10,
                        callbacks=[early_sttoping])


# Evaluar en el conjunto de test
test_loss, test_acc = model.evaluate(test_gen)
print(f'Loss: {test_loss}, test_acc: {test_acc}')

model.save('model.h5')

plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.plot(model_history.history['accuracy'], label='Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.ylabel('Acc')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(model_history.history['loss'], label='Loss')
plt.plot(model_history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('plot.png')
plt.show()


# Prueba con imágenes del conjunto de prueba
class_name = list(test_gen.class_indices.keys())  # Obtener los nombres de las clases

# Generar predicciones para el conjunto de prueba
x_test, y_test = next(test_gen)  # Obtener un lote del generador de prueba
pred = model.predict(x_test)

plt.figure(figsize=(10, 15))
for i in range(9):
    idx = np.random.randint(0, x_test.shape[0])
    plt.subplot(3, 3, i+1)
    plt.yticks([])
    plt.xticks([])
    plt.grid(False)
    
    # Mostrar imagen
    plt.imshow(x_test[idx])
    
    # Etiqueta real y predicción
    real_label = class_name[int(y_test[idx])]
    pred_label = class_name[pred[idx].argmax()]
    
    # Colorear el texto en verde si la predicción es correcta, rojo si es incorrecta
    if y_test[idx] == pred[idx].argmax(): 
        plt.xlabel(f'Real: {real_label}\nPredicción: {pred_label}', color='green')
    else: 
        plt.xlabel(f'Real: {real_label}\nPredicción: {pred_label}', color='red')

plt.show()


y_true = test_gen.classes  # Clases verdaderas de las imágenes del conjunto de prueba
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convierte las probabilidades en clases predichas

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", cm)

# Reporte de clasificación
report = classification_report(y_true, y_pred_classes, target_names=test_gen.class_indices.keys())
print("Classification Report:\n", report)