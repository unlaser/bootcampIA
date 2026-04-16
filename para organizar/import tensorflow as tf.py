import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np 

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    pesos = model.layers[1].get_weights()[0]
    matriz = pesos[:, i].reshape(28, 28)
    axes[i].imshow(matriz, cmap='inferno')
    axes[i].set_title(f'Neurona {i}')
    axes[i].axis('off')
plt.show()

# Cargar el conjunto de datos MNIST (asegurando que las variables estén definidas)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocesar los datos (asegurando que las variables estén definidas)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 3. Exploración de los Datos: Mostrar ejemplos de imágenes
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    # Convertir la etiqueta one-hot de nuevo a un entero para mostrarla
    plt.title(f"Etiqueta: {np.argmax(y_train[i])}")
    plt.axis('off')
plt.suptitle('Ejemplos de dígitos del conjunto de entrenamiento MNIST')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

