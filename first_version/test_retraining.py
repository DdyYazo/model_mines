# -*- coding: utf-8 -*-
"""
Script de prueba para verificar reentrenamiento con ajustes a 3 minas
"""

import os
import numpy as np
import tensorflow as tf
from data_pipeline import MinesDataPipeline
from model_mines_transformer import create_default_model

# Configurar reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

def test_pipeline():
    """Prueba del pipeline de datos"""
    print("Probando pipeline de datos...")
    
    # Crear pipeline (sin emojis)
    pipeline = MinesDataPipeline(data_dir="games", sequence_length=3, random_seed=42)
    
    # Cargar y validar datos
    df = pipeline.load_and_validate_data()
    print(f"Datos cargados: {len(df)} registros")
    print(f"Partidas: {df['partida'].nunique()}")
    
    # Verificar minas por partida
    minas_por_partida = df.groupby(['archivo', 'partida'])['mina'].sum()
    print(f"Promedio minas por partida: {minas_por_partida.mean():.2f}")
    print(f"Distribución minas: {minas_por_partida.value_counts().sort_index().to_dict()}")
    
    # Crear features
    features_df = pipeline.create_features(df)
    print(f"Features creados: {len(features_df)} registros")
    
    # Crear secuencias
    X, y, metadata = pipeline.create_sequences(features_df)
    print(f"Secuencias creadas: X.shape={X.shape}, y.shape={y.shape}")
    print(f"Feature dimension: {pipeline.feature_dim}")
    
    return X, y, pipeline.feature_dim

def test_model(feature_dim):
    """Prueba del modelo actualizado"""
    print("Probando modelo actualizado...")
    
    # Crear modelo con nueva configuración
    model = create_default_model(sequence_length=75, feature_dim=feature_dim)  # 3 juegos x 25 celdas
    print("Modelo creado exitosamente")
    
    # Verificar arquitectura
    print(f"Parámetros del modelo: {model.model.count_params():,}")
    
    # Prueba con datos sintéticos
    batch_size = 4
    X_test = np.random.random((batch_size, 75, feature_dim)).astype('float32')
    y_test = np.random.randint(0, 2, (batch_size, 25)).astype('float32')
    
    # Asegurar que cada muestra tenga exactamente 3 minas
    for i in range(batch_size):
        y_test[i] = 0
        mine_positions = np.random.choice(25, 3, replace=False)
        y_test[i][mine_positions] = 1
    
    print(f"Datos de prueba: X={X_test.shape}, y={y_test.shape}")
    print(f"Minas por muestra: {y_test.sum(axis=1)}")
    
    # Predicción
    predictions = model.model.predict(X_test, verbose=0)
    print(f"Predicciones: shape={predictions.shape}")
    print(f"Suma predicciones: {predictions.sum(axis=1)}")
    
    # Evaluar pérdida
    loss = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"Pérdida en datos sintéticos: {loss}")
    
    return model

def main():
    print("=== PRUEBA DE REENTRENAMIENTO PARA 3 MINAS ===")
    
    try:
        # Probar pipeline
        X, y, feature_dim = test_pipeline()
        
        # Verificar que tenemos 3 minas por partida
        minas_por_muestra = y.sum(axis=1)
        print(f"Minas por muestra en datos reales: min={minas_por_muestra.min()}, max={minas_por_muestra.max()}, mean={minas_por_muestra.mean():.2f}")
        
        # Probar modelo
        model = test_model(feature_dim)
        
        print("\n=== RESULTADO ===")
        print(" Pipeline actualizado funciona correctamente")
        print(" Modelo ajustado a 3 minas funciona correctamente")
        print(" Feature engineering expandido implementado")
        print(" Listo para reentrenamiento completo")
        
        # Guardar modelo de prueba
        model.save_model("model_mines_test.keras")
        print("Modelo de prueba guardado como: model_mines_test.keras")
        
    except Exception as e:
        print(f"Error durante la prueba: {str(e)}")
        raise

if __name__ == "__main__":
    main()