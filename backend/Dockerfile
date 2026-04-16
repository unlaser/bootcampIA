# Utilizamos una imagen de Python oficial compacta (3.11 es muy estable para IA)
FROM python:3.11-slim

# Establecemos el directorio de trabajo dentro de nuestra "máquina virtual" en la nube
WORKDIR /app

# TensorFlow a veces requiere algunas librerías básicas de C++ para funcionar, las instalamos:
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiamos nuestro manifiesto de dependencias al contenedor
COPY requirements.txt .

# Instalamos las librerías de Python sin guardar archivos temporales para ahorrar espacio
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de los archivos de la carpeta backend (main.py, carpeta models, etc)
COPY . .

# Comando de arranque. Cloud Run inyecta automáticamente un puerto (generalmente 8080)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
