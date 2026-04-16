# Flujo de Arquitectura del Sistema

El siguiente diagrama ilustra el viaje completo de la información, desde el navegador del usuario hasta los servidores en la nube de Google y de regreso.

```mermaid
sequenceDiagram
    participant User as Usuario
    participant React as Frontend (React)
    participant FastAPI as Backend (Cloud Run)
    participant Keras as Modelos IA (TensorFlow)

    User->>React: Sube imagen (JPG), digita altura (m) y selecciona género
    Note over User,React: Los datos se guardan en el estado (useState)
    
    User->>React: Clic en "Estimar Medidas"
    React->>React: Empaqueta en FormData (gender, stature_m, file)
    
    React->>FastAPI: Envía Petición HTTP POST (a través de Internet)
    Note over React,FastAPI: FastAPI recibe y extrae las 3 variables
    
    FastAPI->>Keras: Elige Modelo (male/female) y redimensiona Imagen (224x224)
    
    Keras-->>FastAPI: Retorna array matemático crudo (ej. [90.5, 65.2])
    
    FastAPI->>FastAPI: Formatea los datos y construye un diccionario JSON
    
    FastAPI-->>React: Retorna Respuesta (JSON) a través de Internet
    
    React->>React: Extrae predictions.circunferencia_estimada y peso
    React-->>User: Actualiza la pantalla mostrando Peso y Cadera/Abdomen
```

## Nube vs Local
* **Lo que interactúa el usuario (Frontend)**: Corre internamente en su propio navegador usando los recursos de su computadora/móvil (una vez descargada la web de Vercel/Firebase).
* **Lo que procesa los datos (Backend)**: Corre en la memoria de los servidores de **Google Cloud Run**. Allí se hace el trabajo pesado de analizar píxeles mediante las Redes Neuronales, liberando a la computadora del usuario de esta pesada carga computacional.
