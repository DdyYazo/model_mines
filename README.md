# Modelo Transformer para Predicción de Minas GP

Este proyecto implementa un modelo Transformer en Keras/TensorFlow 2.x para predecir las coordenadas de minas en partidas de GP Mines (tablero 5×5).

## Descripción

El modelo predice las 3 posiciones más probables de contener minas en la siguiente partida, basándose en el histórico de partidas anteriores. Utiliza una arquitectura Transformer con codificación posicional y features engineered específicas para el dominio de minas.

## Estructura del Proyecto

```
├── data_pipeline.py          # Pipeline de datos y feature engineering
├── model_mines_transformer.py # Arquitectura del modelo Transformer
├── train.py                  # Script de entrenamiento
├── infer.py                  # Script de inferencia
├── evaluate.py               # Script de evaluación y métricas
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Esta documentación
├── games/                    # Directorio con archivos CSV de datos
│   ├── minas-2025-06-28.csv
│   ├── minas-2025-07-04.csv
│   ├── minas-2025-08-06.csv
│   └── minas-2025-08-07.csv
└── training_logs/            # Logs y reportes de entrenamiento (generado)
```

## Instalación

1. **Clonar o descargar** el proyecto en un directorio local
2. **Crear entorno virtual** (recomendado):
   ```bash
   python -m venv env
   source env/Scripts/activate  # En Windows
   # o
   source env/bin/activate      # En Linux/Mac
   ```
3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### 1. Entrenamiento del Modelo

```bash
python train.py
```

Esto ejecutará:
- Carga y validación de datos CSV
- Feature engineering avanzado
- Entrenamiento del modelo Transformer
- Guardado del modelo entrenado como `model_mines.h5`
- Generación de gráficos de entrenamiento

**Parámetros configurables en `train.py`:**
- `sequence_length`: Número de partidas históricas a usar (default: 3)
- `epochs`: Número máximo de épocas (default: 150)
- `batch_size`: Tamaño de lote (default: 8)
- `patience`: Paciencia para early stopping (default: 25)

### 2. Realizar Predicciones

```bash
python infer.py
```

Esto generará:
- Predicción de las 3 coordenadas más probables para la siguiente partida
- Matriz de probabilidades 5×5
- Análisis de seguridad por filas y columnas
- Visualización de la predicción

**Ejemplo de uso programático:**
```python
from infer import MinesPredictor

predictor = MinesPredictor("model_mines.h5")
result = predictor.predict_next_game(M=3)

print("Top-3 predicciones:", result["top_m_cells"])
print("Filas seguras:", result["safe_rows"])
print("Columnas seguras:", result["safe_cols"])
```

### 3. Evaluación del Modelo

```bash
python evaluate.py
```

Esto generará:
- Métricas Top-K (K=1,2,3,4,5)
- Comparación con baseline de frecuencias históricas
- Análisis de calibración de probabilidades
- Visualizaciones completas
- Reporte JSON con todos los resultados

## Formato de Datos

Los archivos CSV deben tener el formato:
```csv
partida,columna,fila,mina
1,1,1,0
1,1,2,1
...
```

Donde:
- `partida`: ID de la partida (entero)
- `columna`: Coordenada X (1-5)
- `fila`: Coordenada Y (1-5)  
- `mina`: Presencia de mina (0 o 1)

## Arquitectura del Modelo

### Features Engineering
- **Coordenadas normalizadas**: Posición relativa en el tablero
- **Embeddings posicionales**: Codificación sinusoidal de posiciones
- **Features espaciales**: Distancia al centro, bordes, esquinas
- **Features históricos**: Frecuencias acumulativas, tendencias recientes
- **Features de contexto**: Estadísticas por fila/columna

### Modelo Transformer
- **Input**: Secuencias de N partidas × 25 celdas × feature_dim
- **Codificación posicional**: Sinusoidal para secuencias temporales
- **Bloques Transformer**: Multi-Head Attention + Feed Forward
- **Output**: 25 probabilidades (una por celda del tablero)
- **Loss function**: BCE + regularización L2 para suma ≈ 3

### Configuración Optimizada
- **Embed dim**: 24 (reducido para dataset pequeño)
- **Num heads**: 3
- **FF dim**: 48
- **Transformer blocks**: 2
- **Dropout**: 0.3 (alto para regularización)

## Métricas Principales

1. **Top-3 Hit Rate**: Proporción de minas reales encontradas en las 3 predicciones principales
2. **Exact Match Rate**: Proporción de partidas donde se predicen exactamente las 3 minas
3. **Precision@3**: Precisión en las top-3 predicciones
4. **Brier Score**: Calidad de calibración de probabilidades
5. **Mejora vs Baseline**: Comparación con frecuencias históricas

## Reproducibilidad

- **Semilla fija**: `random_seed=42` en todos los componentes
- **Splits temporales**: División estricta por fechas para evitar data leakage
- **Guardado de configuración**: Todos los hiperparámetros se registran

## Limitaciones y Consideraciones

1. **Dataset pequeño**: El modelo está optimizado para pocos datos
2. **Overfitting**: Uso intensivo de dropout y regularización
3. **Calibración**: Las probabilidades pueden requerir recalibración
4. **Drift temporal**: Reentrenamiento periódico recomendado

## Archivos Generados

Después de ejecutar los scripts, se crearán:

- `model_mines.h5`: Modelo entrenado
- `training_curves.png`: Gráficos de entrenamiento  
- `prediction_visualization.png`: Visualización de predicción
- `evaluation_plots/`: Directorio con análisis completo
- `training_logs/`: Reportes JSON detallados
- `evaluation_report.json`: Reporte de evaluación

## Ejemplo de Salida de Predicción

```json
{
  "top_m_cells": [[3,1],[4,5],[2,2]],
  "prob_matrix": [
    [0.04, 0.12, 0.08, 0.15, 0.11],
    [0.09, 0.67, 0.13, 0.07, 0.05],
    [0.85, 0.06, 0.14, 0.22, 0.18],
    [0.11, 0.09, 0.16, 0.08, 0.71],
    [0.07, 0.15, 0.12, 0.19, 0.13]
  ],
  "safe_rows": [2,5],
  "safe_cols": [1,4]
}
```

## Soporte

Para reportar bugs o solicitar features, crear un issue con:
- Descripción del problema
- Archivos de datos utilizados
- Logs de error completos
- Configuración del entorno

## Próximos Pasos

1. **Hyperparameter tuning** con Optuna
2. **Ensemble models** para mayor robustez
3. **Online learning** para adaptación continua
4. **Análisis de drift** temporal de patrones
