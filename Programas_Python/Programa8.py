import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Función para preprocesar imágenes
def preprocess_image(image, target_size=(64, 64)):
    if isinstance(image, str):  # Si es una ruta, carga la imagen
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalizar entre 0 y 1

# Generación de datos de ejemplo
def load_dataset(image_paths, labels, target_size=(64, 64)):
    images = [preprocess_image(img, target_size) for img in image_paths]
    return np.array(images), np.array(labels)

# Rutas a las imágenes y etiquetas
image_paths = [
    'C:\Users\52444\Downloads/Circulo.jpg',
    'C:\Users\52444\Downloads/Triangulo.png',
    'C:\Users\52444\Downloads/Cuadrado.png',
]
labels = [0, 1, 2]  # 0-Triángulo, 1-Cuadrado, 2-Círculo

# Cargar y procesar datos
X, y = load_dataset(image_paths, labels)
y = to_categorical(y, num_classes=3)

# Dividir datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Tres clases: triángulo, cuadrado, círculo
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Guardar el modelo entrenado
model.save('shape_detector_model.h5')

# Función para predecir formas
def predict_shape(image, model, target_size=(64, 64)):
    img = preprocess_image(image, target_size)
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para el batch
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return class_idx

# Cargar modelo
model = tf.keras.models.load_model('shape_detector_model.h5')

# Procesar la imagen de entrada para detectar contornos
image = cv2.imread('C:/Users/Tibs/Documents/python/FigurasColores.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 50, 150)  # Ajusta los umbrales si es necesario
canny = cv2.dilate(canny, None, iterations=1)
canny = cv2.erode(canny, None, iterations=1)
cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Detectar y clasificar formas
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    roi = image[y:y+h, x:x+w]
    if roi.size > 0:  # Validar que la ROI no esté vacía
        class_idx = predict_shape(roi, model)
        label = ["Triángulo", "Cuadrado", "Círculo"][class_idx]
        cv2.putText(image, label, (x, y-5), 1, 1, (0, 255, 0), 1)
    cv2.drawContours(image, [c], 0, (0, 255, 0), 2)

# Mostrar la imagen final
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


