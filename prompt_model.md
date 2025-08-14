# Meta

Eres un **Agente de Ingeniería de Modelos**. Tu objetivo es **diseñar, implementar y validar** un modelo tipo **Transformer** en Keras/TensorFlow 2.x que, dado el histórico de partidas de GP Mines (tablero 5×5), **prediga las coordenadas de las M minas de la siguiente partida** (M por defecto = **3**; si el dataset indica otra cantidad, pregúntame y ajusta). Debes entregar código productizable, reproducible y un informe breve con resultados y comparativas.

# Contexto

* Tablero fijo: **5 columnas × 5 filas**.
* Número de minas por partida: **M = 3** (parametrizable).
* Histórico disponible en CSV a nivel celda con esquema:
  `partida,columna,fila,mina` (mina ∈ {0,1}).
* El histórico puede haber sido registrado con **M≠3** (por ejemplo 4). **Debes detectar y reportar discrepancias** y proponer una adaptación (p.ej., filtrar a sesiones M=3 si existen, o entrenar como clasificación independiente por celda con selección Top-M en inferencia).
* Modelo base de referencia: **Transformer en Keras** con codificación posicional (similar al que se usó para ventas/feriados).
* Objetivo de inferencia: devolver **25 probabilidades** (una por celda), más la **lista de las M coordenadas** con mayor probabilidad de contener mina.
* Debe existir también una vista derivada de **seguridad por fila/columna** (prob. de estar libres) para estrategia de juego.

# Formato de la Respuesta

Responde en **español** y organiza exactamente en estas secciones:

1. **Aclaraciones y Supuestos**

   * Detecta si el CSV cumple 5×5 y consistencia de M por partida.
   * Lista de vacíos/errores de datos y cómo los tratarás.

2. **Plan de Solución (paso a paso)**

   * Pipeline de datos → features → modelo → entrenamiento → evaluación → despliegue.
   * Justifica por qué un Transformer (vs. CNN/LSTM/árboles) y cómo incorporar **restricción Top-M**.

3. **Especificación de Datos & Features**

   * Validación de esquema.
   * Ingeniería de atributos: índices (fila/col), **embeddings posicionales**, conteos históricos por fila/columna, “momentum”/transición entre partidas, ventanas temporales, smoothing y **calibración de probabilidades** (p.ej., Platt/Isotónica).
   * División temporal **rolling origin** (train/val/test) para evitar fuga de información.

4. **Diseño del Modelo (Keras)**

   * Definir **Input** (shape, dtype) y **Output** (25 logits → sigmoide).
   * Capas: `PositionalEncoding`, bloques `MultiHeadAttention`, FFN, normalización, dropout.
   * Hiperparámetros iniciales (embed\_dim, heads, ff\_dim, n\_blocks, dropout).
   * **Pérdida**: BCE por celda + término opcional que fomente **exactamente M** minas (p.ej., penalización L2 a |sum(p)−M|).
   * **Inferencia**: seleccionar **Top-M** celdas por probabilidad; política de desempate.

5. **Entrenamiento y Validación**

   * Optimizador, LR schedule, **early stopping**, batch\_size, épocas.
   * **Búsqueda de hiperparámetros** (simple/Optuna opcional) con espacio acotado.
   * Curvas de pérdida/MAE y registro de seeds para reproducibilidad.

6. **Métricas**

   * **Top-M hit rate** (aciertos/M), **Precision\@M**, **Recall\@M**, **F1\@M**.
   * Brier score y confiabilidad (calibration curve).
   * Baseline comparativo: **mapa de calor histórico** (frecuencia) y regla por filas/columnas.
   * Métrica secundaria: prob. de **fila/columna libre** completa.

7. **Entregables de Código (con nombres de archivo)**

   * `data_pipeline.py`: carga/validación CSV, splits temporales, generación de tensores.
   * `model_mines_transformer.py`: clases `PositionalEncoding`, `TransformerBlock`, constructor del modelo.
   * `train.py`: entrenamiento, checkpoints, early stopping, guardado a `model_mines.h5`.
   * `infer.py`: función `predict_next_game(csv_path, M=3)` → matriz 5×5 de probabilidades + lista de M coordenadas.
   * `evaluate.py`: cálculo de métricas vs. baseline; gráficos PNG (curvas, confiabilidad, heatmaps).
   * `requirements.txt` y `README.md` con pasos para reproducir.
   * (Opcional) `search_hparams.py` para exploración rápida.

8. **Salida Esperada (formato exacto)**

   * **Resumen ejecutivo** (≤150 palabras) con la métrica principal en test.
   * **Bloques de código completos** para cada archivo listado arriba.
   * **Ejemplo de inferencia** (JSON):

     ```json
     {
       "top_m_cells":[[3,1],[4,5],[2,2]],
       "prob_matrix":[[0.04, ... 0.12], ...],
       "safe_rows":[2,5],
       "safe_cols":[1,4]
     }
     ```
   * Enlace local de artefactos generados (modelo `.h5`, gráficos `.png`).

9. **Riesgos y Mitigaciones**

   * Pequeño histórico, drift del RNG, sesgos por overfitting, discrepancia de M.
   * Estrategias: regularización, ensemblado ligero, recalibración periódica, validación “walk-forward”.

10. **Próximos Pasos**

* Cómo reentrenar con nuevas partidas (append → retrain), elección de ventana móvil, alertas de drift.

# Ejemplos

* **Entrada (CSV abreviado)**

  ```
  partida,columna,fila,mina
  1,1,1,0
  1,3,4,1
  ...
  ```
* **Salida (resumen ejecutivo + JSON de inferencia + paths de artefactos)** tal como se especifica en la Sección 8.

# Restricciones

* **Lenguaje**: español.
* **Framework**: **Keras/TensorFlow 2.x** (no PyTorch).
* **No inventes resultados**: si faltan datos o M es inconsistente, **deténte y pregunta**.
* **Reproducibilidad**: fija `random_seed=42`, guarda `model_mines.h5` y logs.
* **Eficiencia**: apuntar a entrenamiento < 5 min en GPU media; inferencia < 50 ms.
* **Entrega**: todo el código debe ejecutarse sin errores en una carpeta limpia; evita rutas absolutas.

# Modelo de guia

Para que entiendas con mas claridad lo que necesito que construyas te comparto un transformer que construi para la prediccion de ventas de un servicios por dia dependiendo de si es un dia festivo o no

```python
# -*- coding: utf-8 -*-
"""Predictor de ingreso prepago v2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NVOxjjBW9_3LZezr-vpncfhvmT3215Sc
"""

#FINAL

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import holidays # Para identificar festivos
import matplotlib.pyplot as plt

# Definir las clases personalizadas para el Transformer
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        
    def call(self, x):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        positions = tf.cast(positions, tf.float32)
        
        # Crear la matriz de codificación posicional
        pos_encoding = np.zeros((self.sequence_length, self.embed_dim))
        for pos in range(self.sequence_length):
            for i in range(0, self.embed_dim, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                if i + 1 < self.embed_dim:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dim)))
        
        pos_encoding = tf.cast(pos_encoding, tf.float32)
        return x + pos_encoding
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim,
        })
        return config

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

df_entrada = pd.read_excel("Revenue Prepago modelo IA.xlsx")
num_days = len(df_entrada)

df_entrada['FECHA'] = pd.to_datetime(df_entrada['FECHA'])
dates = df_entrada['FECHA']

data = []

co_holidays = holidays.CO(years=range(2023, 2023 + int(num_days/365)+1))

for i, date in enumerate(dates):
    day_of_week = date.dayofweek # Lunes=0, Domingo=6
    is_holiday = 1 if date in co_holidays else 0

    amount = df_entrada["REVENUE"][i]
    data.append([date, amount, day_of_week, is_holiday])

df = pd.DataFrame(data, columns=['Date', 'Amount', 'DayOfWeek', 'IsHoliday'])
df['Date'] = pd.to_datetime(df['Date'])
df2 = df.copy()

# One-Hot Encoding for DayOfWeek
df = pd.get_dummies(df, columns=['DayOfWeek'], prefix='DayOfWeek')

# Convert boolean columns to integers (0 or 1)
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

# Scaling the variable to predict
scaler = MinMaxScaler(feature_range=(0, 1))
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])

# Prepare features (X) and target (y)
features_cols = ['Amount_scaled'] + [col for col in df.columns if 'DayOfWeek_' in col] + ['IsHoliday']
X_data = df[features_cols].values
y_data = df['Amount_scaled'].values
timesteps = 7

def create_sequences(input_data, target_data, dates, is_holiday_target, timesteps):
    X, y, target_dates_sequences = [], [], []
    for i in range(len(input_data) - timesteps):
        # Get the sequence of input data
        seq_input = input_data[i:(i + timesteps)]
        # Append the holiday status of the target date to the sequence
        seq_input_with_holiday = np.append(seq_input, np.tile(is_holiday_target[i], (timesteps, 1)), axis=1)
        X.append(seq_input_with_holiday)
        y.append(target_data[i + timesteps])
        target_dates_sequences.append(dates[i + timesteps])
    return np.array(X).astype('float32'), np.array(y).astype('float32'), pd.to_datetime(target_dates_sequences)

def is_colombian_holiday(date):
    """Checks if a given date is a holiday in Colombia."""
    return 1 if date in co_holidays else 0

# Apply the function to the target dates
target_dates = df['Date'][timesteps:]
is_holiday_target = np.array([is_colombian_holiday(date) for date in target_dates])


X_sequences, y_sequences, target_dates_sequences = create_sequences(X_data, y_data, df['Date'], is_holiday_target.reshape(-1, 1), timesteps)

# División de datasets entrenamiento, validación  prueba
# Asegúrate de que los datos de prueba sean posteriores a los de entrenamiento
train_size = int(len(X_sequences) * 0.8)
val_size = int(len(X_sequences) * 0.1)

X_train, y_train = X_sequences[:train_size], y_sequences[:train_size]
X_val, y_val = X_sequences[train_size:train_size + val_size], y_sequences[train_size:train_size + val_size]
X_test, y_test = X_sequences[train_size + val_size:], y_sequences[train_size + val_size:]

# Also split the target dates
target_dates_train = target_dates_sequences[:train_size]
target_dates_val = target_dates_sequences[train_size:train_size + val_size]
target_dates_test = target_dates_sequences[train_size + val_size:]

# Update the embed_dim variable to reflect the new number of features
embed_dim = X_train.shape[2]
print(f"Updated embed_dim: {embed_dim}")

# Modify the inputs layer in the model definition
inputs = layers.Input(shape=(timesteps, embed_dim))
x = inputs

# Ensure that the PositionalEncoding layer is initialized with the new embed_dim
pos_encoding_layer = PositionalEncoding(timesteps, embed_dim)
x = pos_encoding_layer(x)

# Define the Transformer blocks
num_heads = 2 # Changed from 9 to 2
ff_dim = 32
num_transformer_blocks = 7

for _ in range(num_transformer_blocks):
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x, training=True)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation="linear")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# --- 4. Compilación y Entrenamiento ---
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Callback para Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=200, # Puedes aumentar esto
    validation_data=(X_val, y_val),
    # callbacks=[early_stopping]
)

# --- 5. Evaluación ---
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Pérdida (MSE) en conjunto de prueba: {loss:.4f}")
print(f"MAE en conjunto de prueba: {mae:.4f}")

# Predecir y revertir el escalado para interpretar los resultados
predictions_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)
actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

# Ejemplo de visualización (puedes expandir esto)

plt.figure(figsize=(12, 6))

# Plot actuals and predictions for the last 30 days of the test set
plt.plot(target_dates_test[-30:], actuals[-30:], label='Valores Reales')
plt.plot(target_dates_test[-30:], predictions[-30:], label='Predicciones del Modelo')

plt.title('Predicciones del Modelo vs. Valores Reales (Últimos 30 días de prueba)')
plt.xlabel('Fecha')
plt.ylabel('Cantidad de Variable')
plt.legend()

# Formatear el eje X para mostrar las fechas
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%a %d/%m'))
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=2)) # Mostrar cada 2 días para evitar solapamiento
plt.gcf().autofmt_xdate() # Auto-formatear las etiquetas de fecha

# Agregar grilla vertical en las marcas principales
plt.grid(axis='x', color='gray', linestyle='-', linewidth=0.5)

# Establecer los límites del eje y
plt.ylim(0.5e9, 3.5e9)

plt.show()

# Save the model to a file
model.save('model_revenue_prepago_v2.h5')
 
print("Model saved successfully!")
```
es importante que primero analices todos mis archivos y las librerias de mi entorno para de esa manera proceder con la construccion del modelo o transformer
