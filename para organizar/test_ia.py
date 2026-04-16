import tensorflow as tf
import numpy as np
from PIL import Image

print("--- Verificación de Entorno ---")
print(f"Versión de TensorFlow: {tf.__version__}")
print(f"NumPy versión: {np.__version__}")

# Verificar el hardware disponible (CPU/GPU)
dispositivos = tf.config.list_physical_devices()
print(f"Dispositivos detectados: {dispositivos}")

# Realizar una operación básica de TensorFlow
tensor = tf.constant([[1, 2], [3, 4]])
print("\nTensor de prueba exitoso:")
print(tensor.numpy())

# Intentar cargar un modelo descargado de Colab
try:
    # Cargamos el modelo (usamos r'' para evitar problemas con las barras invertidas en Windows)
    modelo = tf.keras.models.load_model(r'bootcampIA\modelo_mnist.keras')
    print("\n¡Modelo de Colab cargado con éxito!")

    # --- Ejemplo de Predicción ---
    # 1. Cargar la imagen real
    # Reemplaza 'tu_imagen.png' por el nombre real de tu archivo
    ruta_imagen = 'bootcampIA/Nuevo Imagen de mapa de bits.png' 
    imagen = Image.open(ruta_imagen).convert('L') # 'L' asegura escala de grises
    img_array = np.array(imagen).astype('float32') / 255.0
    img_ready = np.expand_dims(img_array, axis=0) # Ajustar a forma (1, 28, 28)

    # 2. Realizar la predicción con la imagen real
    predicciones = modelo.predict(img_ready, verbose=0)
    
    # 3. Obtener la clase con mayor probabilidad
    clase_detectada = np.argmax(predicciones)
    print(f"Predicción exitosa. El modelo dice que es la clase: {clase_detectada}")

except Exception as e:
    print(f"\nNo se pudo cargar el modelo: {e}")