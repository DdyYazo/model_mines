# Script especial para predicción inmediata con 3 partidas
import numpy as np
import pandas as pd
from data_pipeline import MinesDataPipeline
from model_mines_transformer import MinesTransformer
import os

def predict_from_3_games():
    """Predice la partida 4 basándose en las 3 partidas proporcionadas"""
    
    # Datos corregidos de las 3 partidas
    csv_content = """partida,columna,fila,mina
1,1,1,0
1,1,2,0
1,1,3,0
1,1,4,0
1,1,5,0
1,2,1,0
1,2,2,0
1,2,3,0
1,2,4,0
1,2,5,1
1,3,1,1
1,3,2,0
1,3,3,0
1,3,4,0
1,3,5,0
1,4,1,0
1,4,2,0
1,4,3,0
1,4,4,1
1,4,5,0
1,5,1,0
1,5,2,0
1,5,3,0
1,5,4,0
1,5,5,0
2,1,1,0
2,1,2,0
2,1,3,0
2,1,4,0
2,1,5,0
2,2,1,0
2,2,2,0
2,2,3,0
2,2,4,0
2,2,5,0
2,3,1,0
2,3,2,0
2,3,3,0
2,3,4,0
2,3,5,1
2,4,1,0
2,4,2,0
2,4,3,0
2,4,4,1
2,4,5,0
2,5,1,0
2,5,2,0
2,5,3,0
2,5,4,1
2,5,5,0
3,1,1,0
3,1,2,0
3,1,3,0
3,1,4,0
3,1,5,0
3,2,1,0
3,2,2,1
3,2,3,0
3,2,4,0
3,2,5,1
3,3,1,0
3,3,2,0
3,3,3,0
3,3,4,0
3,3,5,1
3,4,1,0
3,4,2,0
3,4,3,0
3,4,4,0
3,4,5,0
3,5,1,0
3,5,2,0
3,5,3,0
3,5,4,0
3,5,5,0"""
    
    print("Cargando modelo y preparando datos...")
    
    # Cargar modelo
    model = MinesTransformer.load_model("model_mines_test.keras")
    
    # Crear dataframe
    from io import StringIO
    df = pd.read_csv(StringIO(csv_content))
    df['fecha'] = pd.to_datetime('2025-08-15')
    df['archivo'] = 'current_session.csv'
    
    # Crear pipeline
    pipeline = MinesDataPipeline(sequence_length=3, random_seed=42)
    
    # Generar features manualmente
    print("Generando features...")
    features_df = pipeline.create_features(df)
    
    # Crear secuencia manualmente para las 3 partidas
    # Necesitamos exactamente 75 timesteps (3 partidas × 25 celdas)
    game_data = []
    
    for partida in [1, 2, 3]:
        game_group = features_df[features_df['partida'] == partida].sort_values('pos_index')
        
        if len(game_group) == 25:
            # Seleccionar features numéricas
            feature_cols = [col for col in features_df.columns if col.startswith(('pos_', 'hist_', 'fila_norm', 
                                                                                'col_norm', 'game_', 'row_', 'col_',
                                                                                'neighbors_', 'dist_', 'file_', 'streak_',
                                                                                'cycle_', 'autocorr_'))]
            
            # Asegurar features principales
            required_features = ['pos_index', 'fila_norm', 'col_norm'] + [f'pos_sin_{i}' for i in range(0, 8, 2)] + [f'pos_cos_{i}' for i in range(1, 8, 2)]
            for feat in required_features:
                if feat not in feature_cols:
                    feature_cols.append(feat)
            
            game_features = game_group[feature_cols].values
            game_data.append(game_features)
    
    if len(game_data) == 3:
        # Concatenar las 3 partidas en una secuencia
        X = np.concatenate(game_data, axis=0)  # Shape: (75, n_features)
        X = X.reshape(1, 75, X.shape[1])  # Shape: (1, 75, n_features)
        
        print(f"Datos preparados: {X.shape}")
        print(f"Realizando prediccion...")
        
        # Hacer predicción
        probabilities = model.model.predict(X, verbose=0)[0]
        
        # Procesar resultados
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_coords = []
        top_3_probs = []
        
        for idx in top_3_indices:
            fila = (idx // 5) + 1
            col = (idx % 5) + 1
            top_3_coords.append((col, fila))
            top_3_probs.append(probabilities[idx])
        
        # Análisis de seguridad
        prob_matrix = probabilities.reshape(5, 5)
        row_probs = np.sum(prob_matrix, axis=1)
        col_probs = np.sum(prob_matrix, axis=0)
        
        row_threshold = np.mean(row_probs) - 0.5 * np.std(row_probs)
        col_threshold = np.mean(col_probs) - 0.5 * np.std(col_probs)
        
        safe_rows = [i+1 for i, prob in enumerate(row_probs) if prob < row_threshold]
        safe_cols = [i+1 for i, prob in enumerate(col_probs) if prob < col_threshold]
        
        # Mostrar resultados
        print("\n" + "="*60)
        print("PREDICCION PARA LA PARTIDA 4 DE TU SESION ACTUAL")
        print("="*60)
        
        print("\nHistorial analizado:")
        print("  Partida 1: Minas en (2,5), (3,1), (4,4)")
        print("  Partida 2: Minas en (3,5), (4,4), (5,4)")
        print("  Partida 3: Minas en (2,2), (2,5), (3,5)")
        
        print(f"\nPREDICCION PARA PARTIDA 4:")
        print("Top-3 posiciones recomendadas para buscar minas:")
        
        for i, ((col, fila), prob) in enumerate(zip(top_3_coords, top_3_probs)):
            print(f"  #{i+1}: Columna {col}, Fila {fila}")
            print(f"       Probabilidad: {prob:.1%}")
        
        confidence = np.mean(top_3_probs)
        print(f"\nConfianza general: {confidence:.1%}")
        print(f"Suma total de probabilidades: {np.sum(probabilities):.3f}")
        
        print("\nEstrategia recomendada:")
        print("  Zonas PELIGROSAS (evitar):")
        for col, fila in top_3_coords:
            print(f"    - Columna {col}, Fila {fila}")
        
        print("  Zonas SEGURAS (preferir):")
        print(f"    - Filas mas seguras: {safe_rows}")
        print(f"    - Columnas mas seguras: {safe_cols}")
        
        print("\nMatriz completa de probabilidades:")
        print("     Col1   Col2   Col3   Col4   Col5")
        for i in range(5):
            row_str = f"Fila{i+1}"
            for j in range(5):
                row_str += f" {prob_matrix[i][j]:6.3f}"
            print(row_str)
        
        return {
            'top_3_coords': top_3_coords,
            'top_3_probs': top_3_probs,
            'prob_matrix': prob_matrix,
            'safe_rows': safe_rows,
            'safe_cols': safe_cols,
            'confidence': confidence
        }
    
    else:
        print("Error: No se pudieron procesar las 3 partidas correctamente")
        return None

if __name__ == "__main__":
    try:
        result = predict_from_3_games()
        if result:
            print("\nPrediccion completada exitosamente!")
        else:
            print("\nError en la prediccion.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()