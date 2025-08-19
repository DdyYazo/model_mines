# Predictor inteligente basado en análisis de patrones específicos del usuario
import numpy as np
import pandas as pd
from io import StringIO

class SmartMinesPredictor:
    """Predictor mejorado que aprende de patrones específicos"""
    
    def __init__(self):
        self.user_data = None
        self.pattern_weights = None
        self.position_frequencies = None
        
    def load_user_data(self, csv_data):
        """Carga y analiza los datos del usuario"""
        df = pd.read_csv(StringIO(csv_data))
        self.user_data = df
        self._analyze_patterns()
        
    def _analyze_patterns(self):
        """Analiza patrones específicos en los datos del usuario"""
        print("Analizando patrones específicos del usuario...")
        
        # Calcular frecuencias por posición
        self.position_frequencies = np.zeros((5, 5))
        
        for _, row in self.user_data[self.user_data['mina'] == 1].iterrows():
            self.position_frequencies[row['fila']-1, row['columna']-1] += 1
        
        # Normalizar frecuencias
        total_games = self.user_data['partida'].nunique()
        self.position_frequencies = self.position_frequencies / total_games
        
        # Calcular pesos por patrones detectados
        self.pattern_weights = {
            'column_preference': self._calculate_column_weights(),
            'row_preference': self._calculate_row_weights(),
            'temporal_trend': self._calculate_temporal_weights(),
            'proximity_pattern': self._calculate_proximity_weights()
        }
        
        print("Patrones detectados:")
        print(f"  Preferencia por columnas: {self.pattern_weights['column_preference']}")
        print(f"  Preferencia por filas: {self.pattern_weights['row_preference']}")
        
    def _calculate_column_weights(self):
        """Calcula pesos por columna basado en el historial"""
        col_weights = np.zeros(5)
        for col in range(5):
            col_weights[col] = np.sum(self.position_frequencies[:, col])
        
        # Normalizar
        if np.sum(col_weights) > 0:
            col_weights = col_weights / np.sum(col_weights)
        
        return col_weights
    
    def _calculate_row_weights(self):
        """Calcula pesos por fila basado en el historial"""
        row_weights = np.zeros(5)
        for row in range(5):
            row_weights[row] = np.sum(self.position_frequencies[row, :])
            
        # Normalizar
        if np.sum(row_weights) > 0:
            row_weights = row_weights / np.sum(row_weights)
            
        return row_weights
    
    def _calculate_temporal_weights(self):
        """Calcula tendencias temporales"""
        temporal_weights = {}
        
        # Analizar tendencia por partida
        for partida in sorted(self.user_data['partida'].unique()):
            mines = self.user_data[(self.user_data['partida'] == partida) & (self.user_data['mina'] == 1)]
            avg_col = mines['columna'].mean() if len(mines) > 0 else 0
            avg_row = mines['fila'].mean() if len(mines) > 0 else 0
            temporal_weights[partida] = {'avg_col': avg_col, 'avg_row': avg_row}
        
        return temporal_weights
    
    def _calculate_proximity_weights(self):
        """Calcula patrones de proximidad"""
        # Analizar qué tan cerca aparecen las minas entre sí
        proximities = []
        
        for partida in self.user_data['partida'].unique():
            mines = self.user_data[(self.user_data['partida'] == partida) & (self.user_data['mina'] == 1)]
            coords = [(row['fila'], row['columna']) for _, row in mines.iterrows()]
            
            # Calcular distancias promedio
            if len(coords) >= 2:
                total_dist = 0
                count = 0
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        dist = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                        total_dist += dist
                        count += 1
                
                avg_dist = total_dist / count if count > 0 else 0
                proximities.append(avg_dist)
        
        return np.mean(proximities) if proximities else 2.0
    
    def predict_next_game(self, emphasis_recent=True):
        """Predice la siguiente partida con algoritmo mejorado"""
        
        if self.user_data is None:
            raise ValueError("Datos del usuario no cargados")
        
        print("\nCalculando predicción inteligente...")
        
        # Matriz base de probabilidades
        prob_matrix = self.position_frequencies.copy()
        
        # Factor 1: Frecuencias históricas (peso base)
        base_weight = 0.4
        
        # Factor 2: Tendencia temporal (últimas partidas tienen más peso)
        temporal_weight = 0.3
        last_game = self.user_data['partida'].max()
        recent_games = [last_game-1, last_game] if last_game > 1 else [last_game]
        
        recent_freq = np.zeros((5, 5))
        for partida in recent_games:
            mines = self.user_data[(self.user_data['partida'] == partida) & (self.user_data['mina'] == 1)]
            for _, row in mines.iterrows():
                recent_freq[row['fila']-1, row['columna']-1] += 1
        
        if len(recent_games) > 0:
            recent_freq = recent_freq / len(recent_games)
        
        # Factor 3: Preferencias por columna/fila
        structure_weight = 0.2
        col_prefs = self.pattern_weights['column_preference']
        row_prefs = self.pattern_weights['row_preference']
        
        structure_matrix = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                structure_matrix[i, j] = row_prefs[i] * col_prefs[j]
        
        # Factor 4: Anti-correlación (evitar repetir exactamente la última partida)
        anticorr_weight = 0.1
        last_game_matrix = np.zeros((5, 5))
        last_mines = self.user_data[(self.user_data['partida'] == last_game) & (self.user_data['mina'] == 1)]
        for _, row in last_mines.iterrows():
            last_game_matrix[row['fila']-1, row['columna']-1] = 1
        
        # Combinación ponderada
        final_probs = (base_weight * prob_matrix + 
                      temporal_weight * recent_freq +
                      structure_weight * structure_matrix -
                      anticorr_weight * last_game_matrix)
        
        # Asegurar que no haya probabilidades negativas
        final_probs = np.maximum(final_probs, 0.01)
        
        # Normalizar para que sume aproximadamente 3 (número de minas esperadas)
        total_prob = np.sum(final_probs)
        if total_prob > 0:
            final_probs = final_probs * (3.0 / total_prob)
        
        return final_probs
    
    def get_top_predictions(self, prob_matrix, top_k=3):
        """Obtiene las top-k predicciones"""
        flat_probs = prob_matrix.flatten()
        top_indices = np.argsort(flat_probs)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            fila = (idx // 5) + 1
            col = (idx % 5) + 1
            prob = flat_probs[idx]
            predictions.append((col, fila, prob))
        
        return predictions
    
    def analyze_prediction_quality(self, prob_matrix):
        """Analiza la calidad de la predicción"""
        
        # Calcular concentración en columnas 4-5 (patrón detectado)
        cols_4_5_prob = np.sum(prob_matrix[:, 3:5])
        total_prob = np.sum(prob_matrix)
        concentration_right = cols_4_5_prob / total_prob if total_prob > 0 else 0
        
        # Calcular concentración en filas 4-5 
        rows_4_5_prob = np.sum(prob_matrix[3:5, :])
        concentration_bottom = rows_4_5_prob / total_prob if total_prob > 0 else 0
        
        return {
            'concentration_right_cols': concentration_right,
            'concentration_bottom_rows': concentration_bottom,
            'total_probability': total_prob
        }

def predict_game5_smart():
    """Función principal para predecir la partida 5"""
    
    # Datos del usuario (4 partidas)
    user_data = """partida,columna,fila,mina
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
3,5,5,0
4,1,1,0
4,1,2,0
4,1,3,0
4,1,4,0
4,1,5,0
4,2,1,0
4,2,2,0
4,2,3,0
4,2,4,0
4,2,5,0
4,3,1,0
4,3,2,0
4,3,3,0
4,3,4,0
4,3,5,0
4,4,1,0
4,4,2,0
4,4,3,1
4,4,4,0
4,4,5,0
4,5,1,0
4,5,2,1
4,5,3,1
4,5,4,0
4,5,5,0"""
    
    print("="*60)
    print("PREDICTOR INTELIGENTE V2 - PARTIDA 5")
    print("="*60)
    
    # Crear predictor y cargar datos
    predictor = SmartMinesPredictor()
    predictor.load_user_data(user_data)
    
    # Generar predicción
    prob_matrix = predictor.predict_next_game()
    
    # Obtener top-3 predicciones
    top_predictions = predictor.get_top_predictions(prob_matrix, top_k=3)
    
    # Analizar calidad
    quality = predictor.analyze_prediction_quality(prob_matrix)
    
    # Mostrar resultados
    print(f"\nPREDICCIÓN PARA PARTIDA 5:")
    print("Top-3 posiciones más probables:")
    
    for i, (col, fila, prob) in enumerate(top_predictions):
        print(f"  #{i+1}: Columna {col}, Fila {fila} - Probabilidad: {prob:.1%}")
    
    print(f"\nAnálisis de concentración:")
    print(f"  Concentración en columnas 4-5: {quality['concentration_right_cols']:.1%}")
    print(f"  Concentración en filas 4-5: {quality['concentration_bottom_rows']:.1%}")
    
    # Verificar si sigue el patrón detectado
    if quality['concentration_right_cols'] > 0.5:
        print("  OK Predicción sigue el patrón: preferencia por columnas derechas")
    else:
        print("  AVISO Predicción diverge del patrón detectado")
    
    print("\nMatriz de probabilidades:")
    print("     Col1   Col2   Col3   Col4   Col5")
    for i in range(5):
        row_str = f"Fila{i+1}"
        for j in range(5):
            row_str += f" {prob_matrix[i][j]:6.3f}"
        print(row_str)
    
    # Estrategia recomendada
    print(f"\nESTRATEGIA RECOMENDADA:")
    print("Zonas PELIGROSAS (evitar):")
    for col, fila, prob in top_predictions:
        print(f"  - Columna {col}, Fila {fila} ({prob:.1%})")
    
    # Identificar zonas seguras
    safe_positions = []
    flat_probs = prob_matrix.flatten()
    safe_indices = np.argsort(flat_probs)[:5]  # 5 posiciones más seguras
    
    print("Zonas SEGURAS (preferir):")
    for idx in safe_indices:
        fila = (idx // 5) + 1
        col = (idx % 5) + 1
        prob = flat_probs[idx]
        safe_positions.append((col, fila, prob))
        print(f"  - Columna {col}, Fila {fila} ({prob:.1%})")
    
    return prob_matrix, top_predictions, safe_positions

if __name__ == "__main__":
    prob_matrix, dangerous, safe = predict_game5_smart()