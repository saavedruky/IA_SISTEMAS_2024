import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Desactivar advertencias de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Función para preprocesar imágenes
def preprocess_image(image, target_size=(48, 48)):
    if isinstance(image, str):  # Si es una ruta, cargar la imagen
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # Convertir a escala de grises
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {image}")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalizar entre 0 y 1

# Cargar dataset de expresiones faciales
def load_dataset(image_paths, labels, target_size=(48, 48)):
    images = [preprocess_image(img, target_size) for img in image_paths]
    return np.expand_dims(np.array(images), -1), np.array(labels)

# Rutas de imágenes y etiquetas (modifica según tus datos)
image_paths = [
    'C:/Users/52444/Downloads/Facial/feliz1.png',
    'C:/Users/52444/Downloads/Facial/feliz2.jpg',
    'C:/Users/52444/Downloads/Facial/feliz3.jpg',
    'C:/Users/52444/Downloads/Facial/triste1.png',
    'C:/Users/52444/Downloads/Facial/triste2.png',
    'C:/Users/52444/Downloads/Facial/triste3.png',
    'C:/Users/52444/Downloads/Facial/triste4.png', 
    'C:/Users/52444/Downloads/Facial/triste5.png', 
    'C:/Users/52444/Downloads/Facial/Enojado.png', 
    'C:/Users/52444/Downloads/Facial/Enojado2.png', 
]
labels = [0, 0, 0 , 2, 2, 2, 2, 2, 1, 1]  # Etiquetas: 0-Feliz, 1-Enojado, 2-Triste

X, y = load_dataset(image_paths, labels)
y = to_categorical(y, num_classes=3)  # Codificación one-hot

# Dividir datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear modelo de red neuronal convolucional (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Tres clases: feliz, enojado, triste
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=16)

# Guardar el modelo entrenado
model.save('emotion_detector_model.h5')

# Función para predecir emociones
def predict_emotion(image, model, target_size=(48, 48)):
    img = preprocess_image(image, target_size)
    img = np.expand_dims(img, axis=(0, -1))  # Añadir dimensiones para el batch y canal
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return class_idx

# Cargar el modelo guardado
model = tf.keras.models.load_model('emotion_detector_model.h5')

# Detectar emociones en una imagen con múltiples rostros
image = cv2.imread('C:/Users/52444/Downloads/Facial/Triste5.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

emotion_labels = ["Feliz", "Enojado", "Triste"]

for (x, y, w, h) in faces:
    face = image[y:y+h, x:x+w]
    if face.size > 0:
        class_idx = predict_emotion(face, model)
        label = emotion_labels[class_idx]
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Emotion Detector', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
