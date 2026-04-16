import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Crear la aplicación FastAPI
app = FastAPI(title="Predicción de Métricas Corporales")

# Configurar CORS (Para que el frontend pueda comunicarse sin problemas)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modificar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Calcular el directorio base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Si estamos en Cloud Run (FUSE montado), usar la ruta del bucket. Si no, usar local.
if os.path.exists("/mnt/bucket/modelos"):
    MODELS_DIR = "/mnt/bucket/modelos"
else:
    MODELS_DIR = os.path.join(BASE_DIR, "models")
model_male = None
model_female = None

@app.on_event("startup")
async def load_models():
    """
    Función que se ejecuta al arrancar el servidor.
    Se encarga de cargar los modelos .keras a la memoria.
    """
    global model_male, model_female
    
    # Crear carpeta de modelos si no existe
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    path_male = os.path.join(MODELS_DIR, "modelo_prediccion_male.keras")
    path_female = os.path.join(MODELS_DIR, "modelo_prediccion_female.keras")

    if os.path.exists(path_male):
        model_male = tf.keras.models.load_model(path_male)
        print("✅ Modelo Masculino cargado con éxito.")
    else:
        print(f"⚠️ No se encontró {path_male}")

    if os.path.exists(path_female):
        model_female = tf.keras.models.load_model(path_female)
        print("✅ Modelo Femenino cargado con éxito.")
    else:
        print(f"⚠️ No se encontró {path_female}")


@app.post("/predict")
async def predict(
    gender: str = Form(...),
    stature_m: float = Form(...),
    file: UploadFile = File(...)
):
    """
    Endpoint para realizar la predicción recibiendo una imagen, género y estatura.
    """
    gender = gender.lower().strip()
    if gender not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="El género debe ser 'male' o 'female'.")

    # Seleccionar modelo según el género
    model = model_male if gender == "male" else model_female
    if model is None:
        raise HTTPException(status_code=500, detail=f"El modelo para el género '{gender}' no está disponible o no se cargó correctamente en el backend.")

    # Procesar la imagen
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # Convertir a RGB por si la imagen tiene canal alpha (RGBA)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Redimensionar a 224x224 para ResNet50
        img = img.resize((224, 224))
        
        # Tal como se usaba en Python
        img_array = img_to_array(img)  # Valores en [0, 255]
        img_array = np.expand_dims(img_array, axis=0)  # Agrega la dimensión del batch: (1, 224, 224, 3)
        img_array = preprocess_input(img_array)  # ⚡ BGR + sustracción media ImageNet

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {str(e)}")

    # Preparar el input de estatura en metros (forma N, 1 para Keras)
    try:
        estatura_array = np.array([[stature_m]])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la estatura: {str(e)}")

    # Realizar predicción
    try:
        # El modelo espera [imagen, estatura] tal como está estructurado en Keras con inputs múltiples
        predictions = model.predict([img_array, estatura_array])
        
        # Obtenemos la salida de la lista de arrays
        preds_list = predictions[0].tolist() if isinstance(predictions, np.ndarray) else [p.tolist() for p in predictions]

        # Convertimos la lista cruda en un diccionario más descriptivo
        # Nota: Ajusta los nombres 'circunferencia_abdomen_cadera' y 'peso' según el orden real de tus targets
        pred_dict = {}
        if isinstance(preds_list, list) and len(preds_list) >= 2:
            # TODO: Ajuste artificial temporal (-20% en perímetro y peso) requerido
            pred_dict = {
                "circunferencia_estimada": round(preds_list[0] * 0.80, 4),
                "peso_estimado": round((preds_list[1] * 100) * 0.80, 2)
            }
        else:
            pred_dict = {"resultados": preds_list}

        return {
            "gender": gender,
            "stature_m": stature_m,
            "predictions": pred_dict,
            "message": "Predicción realizada correctamente."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al inferir modelo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
