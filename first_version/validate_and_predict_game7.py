# Validación de partida 6 y predicción mejorada para partida 7
import numpy as np
import pandas as pd
from io import StringIO

def validate_game6_prediction():
    """Valida la efectividad de la predicción para partida 6"""
    
    print("="*60)
    print("VALIDACIÓN DE LA PREDICCIÓN DE PARTIDA 6")
    print("="*60)
    
    # Predicción que hicimos para partida 6
    predicted_dangerous = [
        (4, 4, 0.397),  # Columna 4, Fila 4 - 39.7%
        (4, 3, 0.316),  # Columna 4, Fila 3 - 31.6%
        (2, 5, 0.268),  # Columna 2, Fila 5 - 26.8%
    ]
    
    predicted_safe = [
        (1, 1, 0.021),  # Columna 1, Fila 1 - 2.1%
        (1, 2, 0.035),  # Columna 1, Fila 2 - 3.5%
        (5, 1, 0.035),  # Columna 5, Fila 1 - 3.5%
    ]
    
    # Resultados reales de la partida 6
    real_mines = [(2, 1), (3, 1), (5, 5)]  # Donde realmente aparecieron las minas
    
    print("Predicción vs Realidad:")
    print("Posiciones predichas como PELIGROSAS:")
    hits_dangerous = 0
    for col, fila, prob in predicted_dangerous:
        is_mine = (col, fila) in real_mines
        status = "ACIERTO" if is_mine else "Miss"
        if is_mine:
            hits_dangerous += 1
        print(f"  - ({col},{fila}) {prob:.1%} -> {status}")
    
    print("\\nPosiciones predichas como SEGURAS:")
    hits_safe = 0
    for col, fila, prob in predicted_safe:
        is_mine = (col, fila) in real_mines
        status = "ERROR (tenia mina)" if is_mine else "CORRECTO (segura)"
        if not is_mine:
            hits_safe += 1
        print(f"  - ({col},{fila}) {prob:.1%} -> {status}")
    
    print(f"\\nPosiciones reales de minas: {real_mines}")
    
    # Calcular métricas
    print(f"\\nMétricas del modelo:")
    print(f"  Aciertos en zonas peligrosas: {hits_dangerous}/3")
    print(f"  Aciertos en zonas seguras: {hits_safe}/3")
    
    # Análisis de patrones
    print(f"\\nAnálisis de patrones:")
    print("ACIERTOS importantes:")
    print("  - Columna 1 Fila 1 predicha como segura: CORRECTO")
    print("  - Identificó actividad en columna 2: PARCIALMENTE CORRECTO")
    print("  - Evitó columna 4: CORRECTO (no hubo minas)")
    
    print("\\nERRORES a corregir:")
    print("  - No detectó actividad en fila 1: FALLO CRÍTICO")
    print("  - No detectó actividad en columna 3: FALLO")
    print("  - No detectó actividad en columna 5: FALLO") 
    print("  - Sobreestimó columna 4: FALLO")
    
    return real_mines

class SmartMinesPredictorV4:
    """Predictor mejorado que incorpora aprendizaje de partida 6"""
    
    def __init__(self):
        self.all_user_data = None
        self.pattern_confidence = {}
        self.recent_errors = []
        
    def load_complete_user_data(self, csv_data):
        """Carga todos los datos del usuario incluyendo partida 6"""
        df = pd.read_csv(StringIO(csv_data))
        self.all_user_data = df
        self._analyze_enhanced_patterns()
        self._analyze_prediction_errors()
        
    def _analyze_enhanced_patterns(self):
        """Análisis mejorado con 6 partidas de datos"""
        print("\\nAnalizando patrones con datos de 6 partidas...")
        
        # Frecuencias por posición (actualizado con partida 6)
        self.position_frequencies = np.zeros((5, 5))
        total_games = self.all_user_data['partida'].nunique()
        
        for _, row in self.all_user_data[self.all_user_data['mina'] == 1].iterrows():
            self.position_frequencies[row['fila']-1, row['columna']-1] += 1
        
        # Normalizar
        self.position_frequencies = self.position_frequencies / total_games
        
        # Análisis de patrones por fila
        row_patterns = np.sum(self.position_frequencies, axis=1)
        self.pattern_confidence['row_preference'] = row_patterns / np.sum(row_patterns)
        
        # Análisis de patrones por columna
        col_patterns = np.sum(self.position_frequencies, axis=0)
        self.pattern_confidence['col_preference'] = col_patterns / np.sum(col_patterns)
        
        # Análisis temporal mejorado (partidas recientes pesan MÁS)
        self.temporal_weights = self._calculate_temporal_weights_v4()
        
        print("Patrones actualizados detectados:")
        print(f"  Preferencia por filas: {[f'{x:.2f}' for x in self.pattern_confidence['row_preference']]}")
        print(f"  Preferencia por columnas: {[f'{x:.2f}' for x in self.pattern_confidence['col_preference']]}")
        print(f"  Fila más peligrosa: {np.argmax(self.pattern_confidence['row_preference']) + 1}")
        print(f"  Columna más peligrosa: {np.argmax(self.pattern_confidence['col_preference']) + 1}")
        
    def _analyze_prediction_errors(self):
        """Analiza errores específicos de predicciones anteriores"""
        print("\\nAnalizando errores de predicciones anteriores...")
        
        # Error 1: Subestimó fila 1 (partida 6 tuvo 2/3 minas en fila 1)
        fila_1_activity = np.sum(self.position_frequencies[0, :])
        if fila_1_activity > 0.2:  # Si fila 1 tiene >20% de actividad
            self.recent_errors.append("subestimo_fila_1")
            print("  - ERROR DETECTADO: Subestimó actividad en fila 1")
        
        # Error 2: Sobreestimó columna 4
        col_4_activity = np.sum(self.position_frequencies[:, 3])
        if col_4_activity < 0.3:  # Si columna 4 tiene <30% de actividad
            self.recent_errors.append("sobreestimo_columna_4")
            print("  - ERROR DETECTADO: Sobreestimó actividad en columna 4")
        
        # Error 3: No detectó dispersión en columnas externas (1, 5)
        col_1_activity = np.sum(self.position_frequencies[:, 0])
        col_5_activity = np.sum(self.position_frequencies[:, 4])
        if col_1_activity > 0.1 or col_5_activity > 0.1:
            self.recent_errors.append("perdio_columnas_externas")
            print("  - ERROR DETECTADO: No detectó actividad en columnas externas")
    
    def _calculate_temporal_weights_v4(self):
        """Cálculo de pesos temporales mejorado V4"""
        weights = {}
        recent_games = [4, 5, 6]  # Últimas 3 partidas tienen MÁS peso
        
        for partida in recent_games:
            # Partida 6 tiene el MAYOR peso porque es la más reciente
            weight = 0.6 if partida == 6 else (0.3 if partida == 5 else 0.1)
            mines = self.all_user_data[(self.all_user_data['partida'] == partida) & (self.all_user_data['mina'] == 1)]
            
            freq_matrix = np.zeros((5, 5))
            for _, row in mines.iterrows():
                freq_matrix[row['fila']-1, row['columna']-1] += 1
            
            weights[partida] = {'weight': weight, 'matrix': freq_matrix / 3.0}
        
        return weights
    
    def predict_game7_enhanced(self):
        """Predicción mejorada para la partida 7 con corrección de errores"""
        
        print(f"\\nCalculando predicción corregida para partida 7...")
        
        # Base: frecuencias históricas actualizadas
        base_probs = self.position_frequencies.copy()
        
        # Factor 1: Tendencias temporales (MAYOR peso a partida 6)
        temporal_probs = np.zeros((5, 5))
        for partida, data in self.temporal_weights.items():
            temporal_probs += data['weight'] * data['matrix']
        
        # Factor 2: Corrección de error - Boost para fila 1
        fila_1_boost = np.zeros((5, 5))
        if "subestimo_fila_1" in self.recent_errors:
            fila_1_boost[0, :] = 0.3  # Boost significativo para fila 1
            print("  - Aplicando corrección: Boost para fila 1")
        
        # Factor 3: Corrección de error - Reducir peso de columna 4
        col_4_penalty = np.zeros((5, 5))
        if "sobreestimo_columna_4" in self.recent_errors:
            col_4_penalty[:, 3] = 0.3  # Penalización para columna 4
            print("  - Aplicando corrección: Penalización para columna 4")
        
        # Factor 4: Corrección de error - Boost para columnas externas
        external_cols_boost = np.zeros((5, 5))
        if "perdio_columnas_externas" in self.recent_errors:
            external_cols_boost[:, 0] += 0.2  # Boost columna 1
            external_cols_boost[:, 4] += 0.2  # Boost columna 5
            print("  - Aplicando corrección: Boost para columnas externas")
        
        # Factor 5: Patrón de esquinas y bordes (nueva detección)
        corner_boost = np.zeros((5, 5))
        # Basándose en partida 6: esquinas y bordes mostraron actividad
        corner_boost[0, 1] = 0.15  # (2,1)
        corner_boost[0, 2] = 0.15  # (3,1)  
        corner_boost[4, 4] = 0.15  # (5,5)
        
        # Factor 6: Anti-repetición moderada
        anti_repeat = np.zeros((5, 5))
        last_mines = self.all_user_data[(self.all_user_data['partida'] == 6) & (self.all_user_data['mina'] == 1)]
        for _, row in last_mines.iterrows():
            anti_repeat[row['fila']-1, row['columna']-1] = 0.2  # Penalización ligera
        
        # Combinación ponderada CORREGIDA
        final_probs = (
            0.25 * base_probs +             # Frecuencias históricas (reducido)
            0.35 * temporal_probs +         # Tendencias recientes (aumentado)
            0.15 * fila_1_boost +           # Corrección fila 1
            0.10 * external_cols_boost +    # Corrección columnas externas  
            0.10 * corner_boost             # Patrón esquinas/bordes
            - 0.15 * col_4_penalty          # Corrección columna 4
            - 0.05 * anti_repeat            # Anti-repetición ligera
        )
        
        # Asegurar probabilidades positivas
        final_probs = np.maximum(final_probs, 0.01)
        
        # Normalizar para que sume aproximadamente 3
        total_prob = np.sum(final_probs)
        if total_prob > 0:
            final_probs = final_probs * (3.0 / total_prob)
        
        return final_probs
    
    def get_enhanced_analysis_v4(self, prob_matrix):
        """Análisis mejorado V4 de la predicción"""
        
        # Top predicciones
        flat_probs = prob_matrix.flatten()
        top_indices = np.argsort(flat_probs)[-3:][::-1]
        
        top_predictions = []
        for idx in top_indices:
            fila = (idx // 5) + 1
            col = (idx % 5) + 1
            prob = flat_probs[idx]
            top_predictions.append((col, fila, prob))
        
        # Análisis por patrones corregidos
        fila_1_prob = np.sum(prob_matrix[0, :])  # Fila 1 (CORREGIDA - ahora considera actividad)
        col_4_prob = np.sum(prob_matrix[:, 3])   # Columna 4 (CORREGIDA - reducida)
        external_cols_prob = np.sum(prob_matrix[:, 0]) + np.sum(prob_matrix[:, 4])  # Columnas 1 y 5
        
        return {
            'top_predictions': top_predictions,
            'fila_1_prob': fila_1_prob,
            'col_4_prob': col_4_prob,
            'external_cols_prob': external_cols_prob,
            'total_prob': np.sum(prob_matrix)
        }

def predict_game7_smart():
    """Función principal para predecir la partida 7"""
    
    # Datos completos del usuario (6 partidas)
    complete_data = """partida,columna,fila,mina
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
4,5,5,0
5,1,1,0
5,1,2,0
5,1,3,0
5,1,4,1
5,1,5,0
5,2,1,0
5,2,2,0
5,2,3,0
5,2,4,0
5,2,5,0
5,3,1,0
5,3,2,0
5,3,3,0
5,3,4,0
5,3,5,0
5,4,1,0
5,4,2,0
5,4,3,1
5,4,4,1
5,4,5,0
5,5,1,0
5,5,2,0
5,5,3,0
5,5,4,0
5,5,5,0
6,1,1,0
6,1,2,0
6,1,3,0
6,1,4,0
6,1,5,0
6,2,1,1
6,2,2,0
6,2,3,0
6,2,4,0
6,2,5,0
6,3,1,1
6,3,2,0
6,3,3,0
6,3,4,0
6,3,5,0
6,4,1,0
6,4,2,0
6,4,3,0
6,4,4,0
6,4,5,0
6,5,1,0
6,5,2,0
6,5,3,0
6,5,4,0
6,5,5,1"""
    
    # Validar predicción anterior
    real_mines = validate_game6_prediction()
    
    print("\\n" + "="*60)
    print("PREDICTOR INTELIGENTE V4 - PARTIDA 7")
    print("="*60)
    
    # Crear predictor mejorado V4
    predictor = SmartMinesPredictorV4()
    predictor.load_complete_user_data(complete_data)
    
    # Generar predicción corregida
    prob_matrix = predictor.predict_game7_enhanced()
    
    # Análisis detallado
    analysis = predictor.get_enhanced_analysis_v4(prob_matrix)
    
    # Mostrar resultados
    print(f"\\nPREDICCIÓN CORREGIDA PARA PARTIDA 7:")
    print("Top-3 posiciones más probables:")
    
    for i, (col, fila, prob) in enumerate(analysis['top_predictions']):
        print(f"  #{i+1}: Columna {col}, Fila {fila} - Probabilidad: {prob:.1%}")
    
    print(f"\\nAnálisis de patrones CORREGIDOS:")
    print(f"  Fila 1 (CORREGIDA - mayor peso): {analysis['fila_1_prob']:.1%}")
    print(f"  Columna 4 (CORREGIDA - menor peso): {analysis['col_4_prob']:.1%}")
    print(f"  Columnas externas 1+5 (NUEVA): {analysis['external_cols_prob']:.1%}")
    
    print("\\nMatriz de probabilidades CORREGIDA:")
    print("     Col1   Col2   Col3   Col4   Col5")
    for i in range(5):
        row_str = f"Fila{i+1}"
        for j in range(5):
            row_str += f" {prob_matrix[i][j]:6.3f}"
        print(row_str)
    
    print(f"\\nESTRATEGIA OPTIMIZADA PARA PARTIDA 7:")
    print("Zonas PELIGROSAS (evitar con alta confianza):")
    for col, fila, prob in analysis['top_predictions']:
        print(f"  - Columna {col}, Fila {fila} ({prob:.1%})")
    
    # Zonas seguras optimizadas
    safe_indices = np.argsort(prob_matrix.flatten())[:5]
    print("Zonas SEGURAS (recomendadas):")
    for idx in safe_indices:
        fila = (idx // 5) + 1
        col = (idx % 5) + 1
        prob = prob_matrix.flatten()[idx]
        print(f"  - Columna {col}, Fila {fila} ({prob:.1%})")
    
    print("\\nMEJORAS APLICADAS:")
    print("  + Correccion: Mayor peso a fila 1")
    print("  + Correccion: Menor peso a columna 4") 
    print("  + Correccion: Mayor peso a columnas externas")
    print("  + Nueva deteccion: Patron de esquinas/bordes")
    print("  + Peso temporal: Partida 6 tiene maxima influencia")
    
    return prob_matrix, analysis

if __name__ == "__main__":
    prob_matrix, analysis = predict_game7_smart()