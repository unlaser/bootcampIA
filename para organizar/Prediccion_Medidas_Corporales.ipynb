# %% [markdown]
# # 🏋️ Red Neuronal para Predicción de Medidas Corporales (V2 - Corregido)
# 
# **Objetivo:** Predecir peso (kg) y circunferencia (cadera en mujeres / abdomen en hombres)
# a partir de imágenes de cuerpo completo + estatura como entrada auxiliar.
# 
# **Arquitectura:** ResNet50 (Transfer Learning) con entrada dual (imagen + estatura)
# 
# **Dataset:** 1,000 imágenes sintéticas por género generadas con SMPL-X en Blender
#
# ### Correcciones V2:
# - ⚡ preprocess_input de ResNet50 en vez de /255.0
# - 📸 Data Augmentation (flip, rotación, brillo, contraste)
# - 🧠 Red más profunda (512→256→128) con más dropout
# - 📉 Huber Loss (robusto a outliers) en vez de MSE
# - 🔄 Activación linear en salida (no sigmoid)
# - ⏱️ Más épocas (50+30) con mayor LR inicial

# %% [markdown]
# ## 1. Configuración del Entorno

# %%
# === CONFIGURACIÓN: Cambiar según entorno ===
ENTORNO = "LOCAL"  # Cambiar a "COLAB" si se ejecuta en Google Colab

import os
import warnings
warnings.filterwarnings('ignore')

if ENTORNO == "COLAB":
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = '/content/drive/MyDrive/bootcampIA'  # Ajustar ruta en Drive
    # !pip install -q tensorflow pandas matplotlib scikit-learn
else:
    BASE_PATH = r'c:\Users\Usuario\programacion\bootcampIA'

FEMALE_IMG_DIR = os.path.join(BASE_PATH, 'dataset_female')
MALE_IMG_DIR = os.path.join(BASE_PATH, 'dataset_male')
FEMALE_CSV = os.path.join(BASE_PATH, 'dataset_female', 'database_female', 'database_female.csv')
MALE_CSV = os.path.join(BASE_PATH, 'dataset_male', 'database_male_', 'database_male.csv')

print(f"Entorno: {ENTORNO}")
print(f"Base path: {BASE_PATH}")

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input  # ⚡ CRÍTICO para ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")

# Si hay GPU, limitar memoria para evitar OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# %% [markdown]
# ## 2. Carga y Exploración de Datos

# %%
# Cargar CSVs
df_female = pd.read_csv(FEMALE_CSV)
df_male = pd.read_csv(MALE_CSV)

print("=" * 60)
print("DATASET FEMENINO")
print("=" * 60)
print(f"Shape: {df_female.shape}")
print(df_female.describe().round(3))
print(f"\nColumnas: {list(df_female.columns)}")

print("\n" + "=" * 60)
print("DATASET MASCULINO")
print("=" * 60)
print(f"Shape: {df_male.shape}")
print(df_male.describe().round(3))
print(f"\nColumnas: {list(df_male.columns)}")

# %%
# Visualización de distribuciones
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribución de Variables por Género', fontsize=16, fontweight='bold')

# Femenino
axes[0, 0].hist(df_female['peso_kg'], bins=30, color='#e74c8b', alpha=0.8, edgecolor='white')
axes[0, 0].set_title('Peso - Femenino (kg)')
axes[0, 0].set_xlabel('kg')

axes[0, 1].hist(df_female['perimetro_cadera_m'], bins=30, color='#e74c8b', alpha=0.8, edgecolor='white')
axes[0, 1].set_title('Perímetro Cadera - Femenino (m)')
axes[0, 1].set_xlabel('m')

axes[0, 2].hist(df_female['estatura_m'], bins=30, color='#e74c8b', alpha=0.8, edgecolor='white')
axes[0, 2].set_title('Estatura - Femenino (m)')
axes[0, 2].set_xlabel('m')

# Masculino
axes[1, 0].hist(df_male['peso_kg'], bins=30, color='#3498db', alpha=0.8, edgecolor='white')
axes[1, 0].set_title('Peso - Masculino (kg)')
axes[1, 0].set_xlabel('kg')

axes[1, 1].hist(df_male['perimetro_abdomen_m'], bins=30, color='#3498db', alpha=0.8, edgecolor='white')
axes[1, 1].set_title('Perímetro Abdomen - Masculino (m)')
axes[1, 1].set_xlabel('m')

axes[1, 2].hist(df_male['estatura_m'], bins=30, color='#3498db', alpha=0.8, edgecolor='white')
axes[1, 2].set_title('Estatura - Masculino (m)')
axes[1, 2].set_xlabel('m')

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, 'distribucion_variables.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# Muestra de imágenes con sus medidas
fig, axes = plt.subplots(2, 5, figsize=(20, 9))
fig.suptitle('Muestra de Imágenes con Medidas Ground Truth', fontsize=16, fontweight='bold')

for i in range(5):
    # Femenino
    row = df_female.iloc[i * 50]
    img_path = os.path.join(FEMALE_IMG_DIR, row['image_file'])
    img = load_img(img_path, target_size=(224, 224))
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"♀ {row['peso_kg']:.1f}kg\nCadera: {row['perimetro_cadera_m']:.3f}m\nEst: {row['estatura_m']:.2f}m",
                         fontsize=9)
    axes[0, i].axis('off')

    # Masculino
    row = df_male.iloc[i * 50]
    img_path = os.path.join(MALE_IMG_DIR, row['image_file'])
    img = load_img(img_path, target_size=(224, 224))
    axes[1, i].imshow(img)
    axes[1, i].set_title(f"♂ {row['peso_kg']:.1f}kg\nAbdomen: {row['perimetro_abdomen_m']:.3f}m\nEst: {row['estatura_m']:.2f}m",
                         fontsize=9)
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, 'muestra_imagenes.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. Preprocesamiento de Datos
# 
# ### ⚡ Corrección V2: Usar `preprocess_input` de ResNet50
# ResNet50 fue entrenado con preprocesamiento específico (sustracción de media BGR de ImageNet),
# NO con simple `/255.0`. Usar el preprocesamiento incorrecto hace que las features
# extraídas sean inútiles.

# %%
IMG_SIZE = 224
BATCH_SIZE = 16  # Reducido para máquinas con poca RAM/VRAM

def cargar_imagenes(df, img_dir, target_size=(IMG_SIZE, IMG_SIZE)):
    """Carga imágenes y aplica preprocesamiento CORRECTO de ResNet50."""
    imagenes = []
    indices_validos = []
    for idx, row in df.iterrows():
        img_path = os.path.join(img_dir, row['image_file'])
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)  # Valores en [0, 255]
            img_array = preprocess_input(img_array)  # ⚡ BGR + sustracción media ImageNet
            imagenes.append(img_array)
            indices_validos.append(idx)
        else:
            print(f"⚠️ Imagen no encontrada: {img_path}")
    return np.array(imagenes), indices_validos

print("Cargando imágenes femeninas...")
X_img_female, idx_f = cargar_imagenes(df_female, FEMALE_IMG_DIR)
print(f"  ✅ {len(X_img_female)} imágenes cargadas. Shape: {X_img_female.shape}")

print("Cargando imágenes masculinas...")
X_img_male, idx_m = cargar_imagenes(df_male, MALE_IMG_DIR)
print(f"  ✅ {len(X_img_male)} imágenes cargadas. Shape: {X_img_male.shape}")

# %%
def preparar_datos(df, X_img, indices, col_circunferencia):
    """Prepara inputs (imagen + estatura) y outputs (peso + circunferencia)."""
    df_valid = df.loc[indices].reset_index(drop=True)

    # Input auxiliar: estatura
    X_estatura = df_valid['estatura_m'].values.reshape(-1, 1)

    # Outputs: peso y circunferencia
    y_peso = df_valid['peso_kg'].values.reshape(-1, 1)
    y_circ = df_valid[col_circunferencia].values.reshape(-1, 1)
    Y = np.hstack([y_peso, y_circ])

    # Normalizar estatura y outputs
    scaler_estatura = MinMaxScaler()
    X_estatura_norm = scaler_estatura.fit_transform(X_estatura)

    scaler_y = MinMaxScaler()
    Y_norm = scaler_y.fit_transform(Y)

    # Split: 70% train, 15% val, 15% test
    (X_img_train, X_img_temp, X_est_train, X_est_temp,
     Y_train, Y_temp) = train_test_split(
        X_img, X_estatura_norm, Y_norm, test_size=0.30, random_state=42
    )

    (X_img_val, X_img_test, X_est_val, X_est_test,
     Y_val, Y_test) = train_test_split(
        X_img_temp, X_est_temp, Y_temp, test_size=0.50, random_state=42
    )

    print(f"  Train: {len(X_img_train)} | Val: {len(X_img_val)} | Test: {len(X_img_test)}")

    return {
        'train': ([X_img_train, X_est_train], Y_train),
        'val': ([X_img_val, X_est_val], Y_val),
        'test': ([X_img_test, X_est_test], Y_test),
        'scaler_estatura': scaler_estatura,
        'scaler_y': scaler_y,
    }

print("\n--- Preparando datos FEMENINOS ---")
datos_female = preparar_datos(df_female, X_img_female, idx_f, 'perimetro_cadera_m')

print("\n--- Preparando datos MASCULINOS ---")
datos_male = preparar_datos(df_male, X_img_male, idx_m, 'perimetro_abdomen_m')

# %% [markdown]
# ## 4. Definición de la Arquitectura del Modelo (V2 Mejorada)
#
# ### Mejoras V2:
# - **Data Augmentation** integrada como capas del modelo (solo activas en training)
# - **Red más profunda**: 512→256→128 para imagen, 32→16 para estatura
# - **Huber Loss**: más robusto a outliers que MSE
# - **Activación linear** en salida (sigmoid limitaba la capacidad)
# - **Mayor learning rate** inicial (5e-4 vs 1e-4)

# %%
def crear_modelo(nombre="modelo"):
    """
    Modelo dual-input: Imagen (ResNet50) + Estatura escalar → [peso, circunferencia]
    V2: Con data augmentation, red más profunda y Huber loss
    """
    # --- Rama de imagen: ResNet50 ---
    input_img = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_imagen')

    # Data Augmentation (solo se aplica durante entrenamiento)
    aug = layers.RandomFlip('horizontal')(input_img)
    aug = layers.RandomRotation(0.05)(aug)
    aug = layers.RandomBrightness(0.1)(aug)
    aug = layers.RandomContrast(0.1)(aug)

    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Congelar capas base inicialmente
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model(aug)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', name='dense_img_1')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', name='dense_img_2')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu', name='dense_img_3')(x)

    # --- Rama de estatura ---
    input_estatura = layers.Input(shape=(1,), name='input_estatura')
    y = layers.Dense(32, activation='relu', name='dense_est_1')(input_estatura)
    y = layers.Dense(16, activation='relu', name='dense_est_2')(y)

    # --- Fusión ---
    combined = layers.Concatenate(name='fusion')([x, y])
    z = layers.BatchNormalization()(combined)
    z = layers.Dense(128, activation='relu', name='dense_fusion_1')(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(64, activation='relu', name='dense_fusion_2')(z)
    z = layers.Dense(32, activation='relu', name='dense_fusion_3')(z)

    # --- Salida: [peso_normalizado, circunferencia_normalizada] ---
    output = layers.Dense(2, activation='linear', name='output')(z)
    # linear: permite que la red aprenda valores sin restricción de rango

    model = Model(inputs=[input_img, input_estatura], outputs=output, name=nombre)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss='huber',  # Huber loss: más robusto a outliers que MSE
        metrics=['mae']
    )

    return model, base_model

modelo_female, base_female = crear_modelo("modelo_femenino")
modelo_male, base_male = crear_modelo("modelo_masculino")

print("Modelo Femenino:")
modelo_female.summary()

# %% [markdown]
# ## 5. Entrenamiento - Modelo Femenino (♀)

# %%
EPOCHS_FASE1 = 50  # Más épocas para mejor convergencia
EPOCHS_FASE2 = 30  # Fine-tuning

callbacks_female = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint(os.path.join(BASE_PATH, 'mejor_modelo_female.keras'),
                    monitor='val_loss', save_best_only=True, verbose=1)
]

print("=" * 60)
print("FASE 1 - ENTRENAMIENTO CON CAPAS CONGELADAS (♀)")
print("=" * 60)

history_f1 = modelo_female.fit(
    datos_female['train'][0], datos_female['train'][1],
    validation_data=(datos_female['val'][0], datos_female['val'][1]),
    epochs=EPOCHS_FASE1,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_female,
    verbose=1
)

# %%
# FASE 2: Fine-tuning - descongelar últimas 40 capas de ResNet50
print("=" * 60)
print("FASE 2 - FINE-TUNING ÚLTIMAS CAPAS ResNet50 (♀)")
print("=" * 60)

for layer in base_female.layers[-40:]:
    layer.trainable = True

modelo_female.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # LR más bajo para fine-tuning
    loss='huber',
    metrics=['mae']
)

history_f2 = modelo_female.fit(
    datos_female['train'][0], datos_female['train'][1],
    validation_data=(datos_female['val'][0], datos_female['val'][1]),
    epochs=EPOCHS_FASE2,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_female,
    verbose=1
)

# %% [markdown]
# ## 6. Entrenamiento - Modelo Masculino (♂)

# %%
callbacks_male = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint(os.path.join(BASE_PATH, 'mejor_modelo_male.keras'),
                    monitor='val_loss', save_best_only=True, verbose=1)
]

print("=" * 60)
print("FASE 1 - ENTRENAMIENTO CON CAPAS CONGELADAS (♂)")
print("=" * 60)

history_m1 = modelo_male.fit(
    datos_male['train'][0], datos_male['train'][1],
    validation_data=(datos_male['val'][0], datos_male['val'][1]),
    epochs=EPOCHS_FASE1,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_male,
    verbose=1
)

# %%
# FASE 2: Fine-tuning
print("=" * 60)
print("FASE 2 - FINE-TUNING ÚLTIMAS CAPAS ResNet50 (♂)")
print("=" * 60)

for layer in base_male.layers[-40:]:
    layer.trainable = True

modelo_male.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='huber',
    metrics=['mae']
)

history_m2 = modelo_male.fit(
    datos_male['train'][0], datos_male['train'][1],
    validation_data=(datos_male['val'][0], datos_male['val'][1]),
    epochs=EPOCHS_FASE2,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_male,
    verbose=1
)

# %% [markdown]
# ## 7. Curvas de Entrenamiento

# %%
def plot_training_history(hist1, hist2, titulo, save_name):
    """Grafica loss y MAE para ambas fases de entrenamiento."""
    # Combinar historiales
    loss = hist1.history['loss'] + hist2.history['loss']
    val_loss = hist1.history['val_loss'] + hist2.history['val_loss']
    mae = hist1.history['mae'] + hist2.history['mae']
    val_mae = hist1.history['val_mae'] + hist2.history['val_mae']
    fase1_epochs = len(hist1.history['loss'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(titulo, fontsize=14, fontweight='bold')

    # Loss
    ax1.plot(loss, label='Train Loss', color='#e74c3c')
    ax1.plot(val_loss, label='Val Loss', color='#3498db')
    ax1.axvline(x=fase1_epochs, color='gray', linestyle='--', alpha=0.7, label='Inicio Fine-tuning')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Huber Loss')
    ax1.set_title('Loss (Huber)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MAE
    ax2.plot(mae, label='Train MAE', color='#e74c3c')
    ax2.plot(val_mae, label='Val MAE', color='#3498db')
    ax2.axvline(x=fase1_epochs, color='gray', linestyle='--', alpha=0.7, label='Inicio Fine-tuning')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('MAE')
    ax2.set_title('Error Absoluto Medio (MAE)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, save_name), dpi=150, bbox_inches='tight')
    plt.show()

plot_training_history(history_f1, history_f2, 'Entrenamiento Modelo Femenino (♀)', 'training_female.png')
plot_training_history(history_m1, history_m2, 'Entrenamiento Modelo Masculino (♂)', 'training_male.png')

# %% [markdown]
# ## 8. Evaluación en Conjunto de Test

# %%
def evaluar_modelo(modelo, datos, scaler_y, genero, col_circ_nombre):
    """Evalúa modelo en test set y desnormaliza predicciones."""
    X_test, Y_test_norm = datos['test']

    # Predicciones normalizadas
    Y_pred_norm = modelo.predict(X_test, verbose=0)

    # Desnormalizar
    Y_test_real = scaler_y.inverse_transform(Y_test_norm)
    Y_pred_real = scaler_y.inverse_transform(Y_pred_norm)

    # Separar peso y circunferencia
    peso_real = Y_test_real[:, 0]
    peso_pred = Y_pred_real[:, 0]
    circ_real = Y_test_real[:, 1]
    circ_pred = Y_pred_real[:, 1]

    # Métricas
    resultados = {}
    for nombre, real, pred in [('Peso (kg)', peso_real, peso_pred),
                                (col_circ_nombre, circ_real, circ_pred)]:
        mae = mean_absolute_error(real, pred)
        rmse = np.sqrt(mean_squared_error(real, pred))
        r2 = r2_score(real, pred)
        resultados[nombre] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}

    # Tabla de resultados
    print(f"\n{'=' * 60}")
    print(f"  RESULTADOS - {genero}")
    print(f"{'=' * 60}")
    print(f"{'Métrica':<30} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print(f"{'-' * 60}")
    for nombre, metricas in resultados.items():
        print(f"{nombre:<30} {metricas['MAE']:>10.4f} {metricas['RMSE']:>10.4f} {metricas['R²']:>10.4f}")

    return Y_test_real, Y_pred_real, resultados

print("Evaluando modelo FEMENINO...")
Y_test_f, Y_pred_f, res_f = evaluar_modelo(
    modelo_female, datos_female, datos_female['scaler_y'],
    'MODELO FEMENINO (♀)', 'Perímetro Cadera (m)'
)

print("\nEvaluando modelo MASCULINO...")
Y_test_m, Y_pred_m, res_m = evaluar_modelo(
    modelo_male, datos_male, datos_male['scaler_y'],
    'MODELO MASCULINO (♂)', 'Perímetro Abdomen (m)'
)

# %% [markdown]
# ## 9. Gráficas de Predicción vs Realidad

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Predicción vs Realidad - Conjunto de Test', fontsize=16, fontweight='bold')

# Female - Peso
axes[0, 0].scatter(Y_test_f[:, 0], Y_pred_f[:, 0], alpha=0.6, c='#e74c8b', s=40, edgecolors='white', linewidth=0.5)
lim = [min(Y_test_f[:, 0].min(), Y_pred_f[:, 0].min()) - 5,
       max(Y_test_f[:, 0].max(), Y_pred_f[:, 0].max()) + 5]
axes[0, 0].plot(lim, lim, 'k--', alpha=0.5, label='Predicción perfecta')
axes[0, 0].set_xlabel('Peso Real (kg)')
axes[0, 0].set_ylabel('Peso Predicho (kg)')
axes[0, 0].set_title(f'♀ Peso | R²={res_f["Peso (kg)"]["R²"]:.3f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Female - Cadera
axes[0, 1].scatter(Y_test_f[:, 1], Y_pred_f[:, 1], alpha=0.6, c='#e74c8b', s=40, edgecolors='white', linewidth=0.5)
lim = [min(Y_test_f[:, 1].min(), Y_pred_f[:, 1].min()) - 0.05,
       max(Y_test_f[:, 1].max(), Y_pred_f[:, 1].max()) + 0.05]
axes[0, 1].plot(lim, lim, 'k--', alpha=0.5, label='Predicción perfecta')
axes[0, 1].set_xlabel('Perímetro Cadera Real (m)')
axes[0, 1].set_ylabel('Perímetro Cadera Predicho (m)')
axes[0, 1].set_title(f'♀ Cadera | R²={res_f["Perímetro Cadera (m)"]["R²"]:.3f}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Male - Peso
axes[1, 0].scatter(Y_test_m[:, 0], Y_pred_m[:, 0], alpha=0.6, c='#3498db', s=40, edgecolors='white', linewidth=0.5)
lim = [min(Y_test_m[:, 0].min(), Y_pred_m[:, 0].min()) - 5,
       max(Y_test_m[:, 0].max(), Y_pred_m[:, 0].max()) + 5]
axes[1, 0].plot(lim, lim, 'k--', alpha=0.5, label='Predicción perfecta')
axes[1, 0].set_xlabel('Peso Real (kg)')
axes[1, 0].set_ylabel('Peso Predicho (kg)')
axes[1, 0].set_title(f'♂ Peso | R²={res_m["Peso (kg)"]["R²"]:.3f}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Male - Abdomen
axes[1, 1].scatter(Y_test_m[:, 1], Y_pred_m[:, 1], alpha=0.6, c='#3498db', s=40, edgecolors='white', linewidth=0.5)
lim = [min(Y_test_m[:, 1].min(), Y_pred_m[:, 1].min()) - 0.05,
       max(Y_test_m[:, 1].max(), Y_pred_m[:, 1].max()) + 0.05]
axes[1, 1].plot(lim, lim, 'k--', alpha=0.5, label='Predicción perfecta')
axes[1, 1].set_xlabel('Perímetro Abdomen Real (m)')
axes[1, 1].set_ylabel('Perímetro Abdomen Predicho (m)')
axes[1, 1].set_title(f'♂ Abdomen | R²={res_m["Perímetro Abdomen (m)"]["R²"]:.3f}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, 'prediccion_vs_realidad.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 10. Visualización de Predicciones en Imágenes

# %%
def mostrar_predicciones(modelo, datos, scaler_y, img_dir, df, indices,
                         genero, col_circ, col_circ_nombre, n=8):
    """Muestra imágenes del test set con predicción vs ground truth."""
    X_test, Y_test_norm = datos['test']
    Y_pred_norm = modelo.predict(X_test, verbose=0)

    Y_test_real = scaler_y.inverse_transform(Y_test_norm)
    Y_pred_real = scaler_y.inverse_transform(Y_pred_norm)

    n = min(n, len(X_test[0]))
    fig, axes = plt.subplots(2, n // 2, figsize=(4 * (n // 2), 10))
    fig.suptitle(f'Predicciones {genero} (Verde=Real, Rojo=Predicción)', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i in range(n):
        # Las imagenes están preprocesadas, hay que revertir para visualizar
        img_display = X_test[0][i].copy()
        # Revertir preprocess_input (aproximado): sumar medias BGR y convertir a RGB
        img_display[:, :, 0] += 103.939
        img_display[:, :, 1] += 116.779
        img_display[:, :, 2] += 123.68
        img_display = img_display[:, :, ::-1]  # BGR → RGB
        img_display = np.clip(img_display / 255.0, 0, 1)

        axes[i].imshow(img_display)
        real_p, pred_p = Y_test_real[i, 0], Y_pred_real[i, 0]
        real_c, pred_c = Y_test_real[i, 1], Y_pred_real[i, 1]

        texto = (f"Peso: {real_p:.1f}→{pred_p:.1f}kg\n"
                 f"{col_circ_nombre}: {real_c:.3f}→{pred_c:.3f}m")
        axes[i].set_title(texto, fontsize=8)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, f'predicciones_{genero.lower()}.png'), dpi=150, bbox_inches='tight')
    plt.show()

mostrar_predicciones(modelo_female, datos_female, datos_female['scaler_y'],
                     FEMALE_IMG_DIR, df_female, idx_f, 'Femenino',
                     'perimetro_cadera_m', 'Cadera')

mostrar_predicciones(modelo_male, datos_male, datos_male['scaler_y'],
                     MALE_IMG_DIR, df_male, idx_m, 'Masculino',
                     'perimetro_abdomen_m', 'Abdomen')

# %% [markdown]
# ## 11. Guardado de Modelos y Scalers

# %%
# Guardar modelos
modelo_female.save(os.path.join(BASE_PATH, 'modelo_prediccion_female.keras'))
modelo_male.save(os.path.join(BASE_PATH, 'modelo_prediccion_male.keras'))

# Guardar scalers para uso posterior
import pickle
scalers = {
    'female': {
        'scaler_estatura': datos_female['scaler_estatura'],
        'scaler_y': datos_female['scaler_y'],
    },
    'male': {
        'scaler_estatura': datos_male['scaler_estatura'],
        'scaler_y': datos_male['scaler_y'],
    }
}
with open(os.path.join(BASE_PATH, 'scalers_prediccion.pkl'), 'wb') as f:
    pickle.dump(scalers, f)

print("✅ Modelos guardados:")
print(f"   - {os.path.join(BASE_PATH, 'modelo_prediccion_female.keras')}")
print(f"   - {os.path.join(BASE_PATH, 'modelo_prediccion_male.keras')}")
print(f"   - {os.path.join(BASE_PATH, 'scalers_prediccion.pkl')}")

# %% [markdown]
# ## 12. Resumen Final

# %%
print("=" * 70)
print("  RESUMEN DE RESULTADOS - PREDICCIÓN DE MEDIDAS CORPORALES")
print("=" * 70)

print(f"\n{'Modelo':<20} {'Variable':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
print("-" * 75)
for nombre, metricas in res_f.items():
    print(f"{'Femenino (♀)':<20} {nombre:<25} {metricas['MAE']:>10.4f} {metricas['RMSE']:>10.4f} {metricas['R²']:>10.4f}")
for nombre, metricas in res_m.items():
    print(f"{'Masculino (♂)':<20} {nombre:<25} {metricas['MAE']:>10.4f} {metricas['RMSE']:>10.4f} {metricas['R²']:>10.4f}")

print("\n" + "=" * 70)
print("  Arquitectura: ResNet50 + Estatura (dual-input) → [Peso, Circunferencia]")
print(f"  Dataset: {len(df_female)} mujeres + {len(df_male)} hombres")
print(f"  Entrenamiento: {EPOCHS_FASE1} épocas (congelado) + {EPOCHS_FASE2} épocas (fine-tuning)")
print("  Loss: Huber | Preprocesamiento: preprocess_input (BGR + media ImageNet)")
print("=" * 70)
