import pandas as pd
import numpy as np
from collections import defaultdict, Counter

class EnhancedMinesPredictorV6:
    """Predictor rediseñado basado en análisis de fallos anteriores"""
    
    def __init__(self, csv_file_path):
        self.csv_file = csv_file_path
        self.load_data()
        self.position_frequencies = None
        self.pattern_weights = {}
        self.error_corrections = []
        
    def load_data(self):
        """Carga datos con análisis dinámico de partidas disponibles"""
        df = pd.read_csv(self.csv_file)
        
        # Organizar por partidas
        self.all_games = {}
        for _, row in df.iterrows():
            game_id = int(row['partida'])
            if game_id not in self.all_games:
                self.all_games[game_id] = []
            self.all_games[game_id].append({
                'col': int(row['columna']),
                'row': int(row['fila']), 
                'mine': int(row['mina'])
            })
        
        # Usar las últimas 3 partidas disponibles dinámicamente
        sorted_games = sorted(self.all_games.keys())
        self.last_games = sorted_games[-3:] if len(sorted_games) >= 3 else sorted_games
        self.next_game_number = max(sorted_games) + 1 if sorted_games else 1
        self.total_games = len(sorted_games)
        
        print(f"Datos cargados: {self.total_games} partidas completadas")
        print(f"Analizando ultimas 3 partidas: {self.last_games}")
        
    def analyze_comprehensive_patterns(self):
        """Análisis comprehensivo mejorado de patrones"""
        print(f"\n=== ANALISIS COMPREHENSIVO DE PATRONES ===")
        
        # 1. Frecuencias históricas globales
        self.position_frequencies = np.zeros((5, 5))
        all_mine_positions = []
        
        for game_id in self.all_games:
            for pos in self.all_games[game_id]:
                if pos['mine'] == 1:
                    self.position_frequencies[pos['row']-1, pos['col']-1] += 1
                    all_mine_positions.append((pos['col'], pos['row']))
        
        # Normalizar por número de partidas
        if self.total_games > 0:
            self.position_frequencies = self.position_frequencies / self.total_games
        
        # 2. Análisis temporal con peso exponencial
        temporal_matrix = np.zeros((5, 5))
        weights = [0.5, 0.3, 0.2]  # Última partida tiene más peso
        
        for i, game_id in enumerate(reversed(self.last_games)):
            weight = weights[i] if i < len(weights) else 0.1
            for pos in self.all_games[game_id]:
                if pos['mine'] == 1:
                    temporal_matrix[pos['row']-1, pos['col']-1] += weight
        
        self.temporal_matrix = temporal_matrix
        
        # 3. Análisis de dispersión dinámico
        self.dispersion_patterns = self._analyze_dispersion_patterns()
        
        # 4. Análisis de columnas/filas emergentes
        self.emerging_patterns = self._detect_emerging_patterns()
        
        # 5. Análisis de anti-clustering
        self.clustering_analysis = self._analyze_clustering_tendencies()
        
        return {
            'historical': self.position_frequencies,
            'temporal': temporal_matrix,
            'dispersion': self.dispersion_patterns,
            'emerging': self.emerging_patterns,
            'clustering': self.clustering_analysis
        }
    
    def _analyze_dispersion_patterns(self):
        """Analiza patrones de dispersión en las últimas partidas"""
        dispersion_scores = []
        
        for game_id in self.last_games:
            mines = []
            for pos in self.all_games[game_id]:
                if pos['mine'] == 1:
                    mines.append((pos['col'], pos['row']))
            
            if len(mines) == 3:
                # Calcular dispersión Manhattan
                distances = []
                for i in range(3):
                    for j in range(i+1, 3):
                        dist = abs(mines[i][0] - mines[j][0]) + abs(mines[i][1] - mines[j][1])
                        distances.append(dist)
                avg_distance = sum(distances) / len(distances)
                dispersion_scores.append(avg_distance)
        
        avg_dispersion = sum(dispersion_scores) / len(dispersion_scores) if dispersion_scores else 3.0
        
        return {
            'avg_dispersion': avg_dispersion,
            'tendency': 'disperso' if avg_dispersion > 4 else ('moderado' if avg_dispersion > 2.5 else 'agrupado'),
            'recent_scores': dispersion_scores
        }
    
    def _detect_emerging_patterns(self):
        """Detecta patrones emergentes en columnas/filas"""
        if self.total_games < 3:
            return {'emerging_cols': [], 'emerging_rows': [], 'declining_areas': []}
        
        # Comparar actividad en últimas 2 vs anteriores
        recent_activity = defaultdict(int)
        early_activity = defaultdict(int)
        
        recent_games = self.last_games[-2:] if len(self.last_games) >= 2 else self.last_games
        early_games = [g for g in self.all_games.keys() if g not in recent_games]
        
        # Actividad reciente
        for game_id in recent_games:
            for pos in self.all_games[game_id]:
                if pos['mine'] == 1:
                    recent_activity[f"col_{pos['col']}"] += 1
                    recent_activity[f"row_{pos['row']}"] += 1
        
        # Actividad temprana
        for game_id in early_games:
            for pos in self.all_games[game_id]:
                if pos['mine'] == 1:
                    early_activity[f"col_{pos['col']}"] += 1
                    early_activity[f"row_{pos['row']}"] += 1
        
        emerging_cols = []
        emerging_rows = []
        declining_areas = []
        
        # Detectar emergentes (más actividad reciente que histórica)
        for area in recent_activity:
            recent_rate = recent_activity[area] / len(recent_games) if recent_games else 0
            early_rate = early_activity[area] / len(early_games) if early_games else 0
            
            if recent_rate > early_rate * 1.5:  # 50% más activo recientemente
                if area.startswith('col'):
                    emerging_cols.append(int(area.split('_')[1]))
                else:
                    emerging_rows.append(int(area.split('_')[1]))
            elif early_rate > recent_rate * 1.5:  # Declive
                declining_areas.append(area)
        
        return {
            'emerging_cols': emerging_cols,
            'emerging_rows': emerging_rows,
            'declining_areas': declining_areas
        }
    
    def _analyze_clustering_tendencies(self):
        """Analiza tendencias de agrupamiento vs dispersión"""
        cluster_analysis = []
        
        for game_id in self.last_games:
            mines = []
            for pos in self.all_games[game_id]:
                if pos['mine'] == 1:
                    mines.append((pos['col'], pos['row']))
            
            if len(mines) == 3:
                # Verificar si hay clustering
                adjacency_count = 0
                for i, mine1 in enumerate(mines):
                    for j, mine2 in enumerate(mines):
                        if i != j:
                            dist = abs(mine1[0] - mine2[0]) + abs(mine1[1] - mine2[1])
                            if dist <= 2:  # Adyacentes o con una casilla de separación
                                adjacency_count += 1
                
                cluster_score = adjacency_count / 6  # Normalizar (máximo 6 pares)
                cluster_analysis.append(cluster_score)
        
        avg_clustering = sum(cluster_analysis) / len(cluster_analysis) if cluster_analysis else 0.0
        
        return {
            'avg_clustering': avg_clustering,
            'anti_cluster_tendency': avg_clustering < 0.3,  # Si es bajo, tiende a dispersar
            'recent_scores': cluster_analysis
        }
    
    def calculate_enhanced_probabilities(self):
        """Calcula probabilidades mejoradas usando todos los análisis"""
        patterns = self.analyze_comprehensive_patterns()
        
        # Matriz base
        prob_matrix = np.ones((5, 5)) * 0.02  # Base mínima muy baja
        
        # Factor 1: Frecuencias históricas (20% peso - reducido)
        prob_matrix += 0.2 * patterns['historical']
        
        # Factor 2: Tendencias temporales (30% peso - aumentado)
        prob_matrix += 0.3 * patterns['temporal']
        
        # Factor 3: Corrección por dispersión
        if patterns['dispersion']['tendency'] == 'disperso':
            # Boost esquinas y bordes
            prob_matrix[0, :] += 0.1  # Fila 1
            prob_matrix[4, :] += 0.1  # Fila 5
            prob_matrix[:, 0] += 0.1  # Columna 1
            prob_matrix[:, 4] += 0.1  # Columna 5
            prob_matrix[2, 2] -= 0.05  # Reducir centro
        elif patterns['dispersion']['tendency'] == 'agrupado':
            # Boost centro
            prob_matrix[1:4, 1:4] += 0.1
        
        # Factor 4: Patrones emergentes (CRÍTICO - aquí estaba el error)
        for col in patterns['emerging']['emerging_cols']:
            prob_matrix[:, col-1] += 0.2  # Boost significativo
            print(f"  Aplicando boost a columna emergente {col}")
        
        for row in patterns['emerging']['emerging_rows']:
            prob_matrix[row-1, :] += 0.2  # Boost significativo
            print(f"  Aplicando boost a fila emergente {row}")
        
        # Factor 5: Anti-clustering si es la tendencia
        if patterns['clustering']['anti_cluster_tendency']:
            # Penalizar posiciones adyacentes a minas recientes
            for game_id in self.last_games[-1:]:  # Solo última partida
                for pos in self.all_games[game_id]:
                    if pos['mine'] == 1:
                        # Penalizar vecindad
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = pos['row']-1+dr, pos['col']-1+dc
                                if 0 <= nr < 5 and 0 <= nc < 5:
                                    prob_matrix[nr, nc] -= 0.05
        
        # Factor 6: Corrección específica por fallos anteriores
        self._apply_failure_corrections(prob_matrix)
        
        # Normalizar
        prob_matrix = np.maximum(prob_matrix, 0.005)  # Mínimo 0.5%
        total_prob = np.sum(prob_matrix)
        if total_prob > 0:
            prob_matrix = prob_matrix * (3.0 / total_prob)
        
        return prob_matrix
    
    def _apply_failure_corrections(self, prob_matrix):
        """Aplica correcciones específicas basadas en fallos anteriores"""
        print(f"\nAplicando correcciones basadas en fallos anteriores...")
        
        # Corrección 1: No obsesionarse con una sola columna/fila
        max_col_prob = np.max(np.sum(prob_matrix, axis=0))
        max_row_prob = np.max(np.sum(prob_matrix, axis=1))
        
        if max_col_prob > 1.5:  # Si una columna tiene >50% probabilidad total
            max_col = np.argmax(np.sum(prob_matrix, axis=0))
            prob_matrix[:, max_col] *= 0.7  # Reducir 30%
            print(f"  Reduciendo obsesion con columna {max_col+1}")
        
        if max_row_prob > 1.5:  # Si una fila tiene >50% probabilidad total
            max_row = np.argmax(np.sum(prob_matrix, axis=1))
            prob_matrix[max_row, :] *= 0.7  # Reducir 30%
            print(f"  Reduciendo obsesion con fila {max_row+1}")
        
        # Corrección 2: Boost para zonas sistemáticamente ignoradas
        # Identificar posiciones que han aparecido pero el modelo subestima
        for i in range(5):
            for j in range(5):
                historical_freq = self.position_frequencies[i, j]
                current_prob = prob_matrix[i, j]
                
                # Si la posición ha aparecido históricamente pero está subestimada
                if historical_freq > 0.15 and current_prob < historical_freq * 0.5:
                    prob_matrix[i, j] += 0.1
                    print(f"  Corrigiendo subestimacion en ({j+1},{i+1})")
    
    def get_enhanced_predictions(self, prob_matrix):
        """Obtiene predicciones mejoradas con justificaciones detalladas"""
        flat_probs = prob_matrix.flatten()
        top_indices = np.argsort(flat_probs)[::-1][:3]
        
        predictions = []
        for i, idx in enumerate(top_indices):
            row = idx // 5 + 1
            col = idx % 5 + 1
            prob = flat_probs[idx] * 100
            
            # Justificación mejorada
            justifications = []
            
            # Frecuencia histórica
            hist_freq = self.position_frequencies[row-1, col-1]
            if hist_freq > 0.1:
                justifications.append(f"Frecuencia historica {hist_freq:.1%}")
            
            # Actividad temporal
            temporal_activity = self.temporal_matrix[row-1, col-1]
            if temporal_activity > 0.1:
                justifications.append(f"Alta actividad reciente")
            
            # Patrones emergentes
            if col in self.emerging_patterns.get('emerging_cols', []):
                justifications.append(f"Columna {col} emergente")
            if row in self.emerging_patterns.get('emerging_rows', []):
                justifications.append(f"Fila {row} emergente")
            
            # Posición estratégica
            if row in [1, 5] or col in [1, 5]:
                justifications.append("Posicion de borde")
            if (row == 1 and col == 1) or (row == 1 and col == 5) or (row == 5 and col == 1) or (row == 5 and col == 5):
                justifications.append("Esquina estrategica")
            
            # Patrón de dispersión
            if self.dispersion_patterns['tendency'] == 'disperso' and (row in [1, 5] or col in [1, 5]):
                justifications.append("Patron de dispersion detectado")
            
            predictions.append({
                'position': (col, row),
                'probability': prob,
                'justification': " + ".join(justifications) if justifications else "Analisis estadistico combinado"
            })
        
        return predictions
    
    def predict_next_game_enhanced(self):
        """Predicción mejorada de la siguiente partida"""
        print("PREDICTOR MEJORADO V6 - BASADO EN ANALISIS DE FALLOS")
        print("=" * 70)
        
        print(f"\nPREDICCION PARA PARTIDA {self.next_game_number}")
        print(f"Analizando ultimas partidas: {self.last_games}")
        
        # Calcular probabilidades mejoradas
        prob_matrix = self.calculate_enhanced_probabilities()
        
        # Obtener predicciones
        predictions = self.get_enhanced_predictions(prob_matrix)
        
        # Análisis detallado de patrones detectados
        print(f"\n=== PATRONES DETECTADOS ===")
        print(f"Dispersion promedio: {self.dispersion_patterns['avg_dispersion']:.1f} - {self.dispersion_patterns['tendency']}")
        
        if self.emerging_patterns['emerging_cols']:
            print(f"Columnas emergentes: {self.emerging_patterns['emerging_cols']}")
        if self.emerging_patterns['emerging_rows']:
            print(f"Filas emergentes: {self.emerging_patterns['emerging_rows']}")
        
        if self.clustering_analysis['anti_cluster_tendency']:
            print("Tendencia anti-clustering detectada")
        
        # Mostrar predicciones
        print(f"\nPREDICCIONES TOP 3 PARA PARTIDA {self.next_game_number}:")
        for i, pred in enumerate(predictions, 1):
            pos = pred['position']
            prob = pred['probability']
            just = pred['justification']
            print(f"\nMINA #{i}: ({pos[0]},{pos[1]}) - {prob:.1f}% probabilidad")
            print(f"  Justificacion: {just}")
        
        # Análisis de seguridad mejorado
        safe_zones = self._get_enhanced_safe_zones(prob_matrix)
        
        print(f"\n=== ZONAS SEGURAS RECOMENDADAS ===")
        for i, safe in enumerate(safe_zones[:5], 1):
            pos = safe['position']
            prob = safe['probability']
            safety = safe['safety']
            reasoning = safe.get('reasoning', '')
            print(f"{i}. ({pos[0]},{pos[1]}): {prob:.1f}% riesgo - {safety}")
            if reasoning:
                print(f"   Razon: {reasoning}")
        
        # Estrategia específica
        print(f"\n=== ESTRATEGIA RECOMENDADA ===")
        self._generate_strategy_recommendations(predictions, safe_zones, prob_matrix)
        
        return predictions, safe_zones, prob_matrix
    
    def _get_enhanced_safe_zones(self, prob_matrix):
        """Identifica zonas seguras con análisis mejorado"""
        safe_positions = []
        
        for row in range(5):
            for col in range(5):
                prob = prob_matrix[row, col] * 100
                
                if prob < 8.0:  # Considerablemente seguras
                    # Clasificar nivel de seguridad
                    if prob < 2.0:
                        safety = 'ULTRA-SEGURA'
                    elif prob < 4.0:
                        safety = 'MUY SEGURA'
                    elif prob < 6.0:
                        safety = 'SEGURA'
                    else:
                        safety = 'MODERADAMENTE SEGURA'
                    
                    # Análisis de por qué es segura
                    reasoning = []
                    
                    # Si está en zona históricamente poco activa
                    hist_freq = self.position_frequencies[row, col]
                    if hist_freq < 0.1:
                        reasoning.append("historicamente poco activa")
                    
                    # Si está alejada de patrones emergentes
                    col_num = col + 1
                    row_num = row + 1
                    if col_num not in self.emerging_patterns.get('emerging_cols', []) and row_num not in self.emerging_patterns.get('emerging_rows', []):
                        reasoning.append("alejada de patrones emergentes")
                    
                    # Si está en zona central y la tendencia es dispersa
                    if self.dispersion_patterns['tendency'] == 'disperso' and 1 <= row <= 3 and 1 <= col <= 3:
                        reasoning.append("zona central con tendencia dispersa")
                    
                    safe_positions.append({
                        'position': (col+1, row+1),
                        'probability': prob,
                        'safety': safety,
                        'reasoning': ", ".join(reasoning) if reasoning else "analisis estadistico"
                    })
        
        return sorted(safe_positions, key=lambda x: x['probability'])
    
    def _generate_strategy_recommendations(self, predictions, safe_zones, prob_matrix):
        """Genera recomendaciones estratégicas específicas"""
        # Análisis por filas y columnas
        row_risks = [np.sum(prob_matrix[i, :]) * 100 for i in range(5)]
        col_risks = [np.sum(prob_matrix[:, j]) * 100 for j in range(5)]
        
        safest_row = np.argmin(row_risks) + 1
        safest_col = np.argmin(col_risks) + 1
        most_dangerous_row = np.argmax(row_risks) + 1
        most_dangerous_col = np.argmax(col_risks) + 1
        
        print(f"FILA MAS SEGURA: {safest_row} ({row_risks[safest_row-1]:.1f}% riesgo total)")
        print(f"COLUMNA MAS SEGURA: {safest_col} ({col_risks[safest_col-1]:.1f}% riesgo total)")
        print(f"\nEVITAR:")
        print(f"  - Fila {most_dangerous_row} ({row_risks[most_dangerous_row-1]:.1f}% riesgo)")
        print(f"  - Columna {most_dangerous_col} ({col_risks[most_dangerous_col-1]:.1f}% riesgo)")
        
        # Recomendación de primera jugada
        if safe_zones:
            best_safe = safe_zones[0]
            print(f"\nPRIMERA JUGADA RECOMENDADA: ({best_safe['position'][0]},{best_safe['position'][1]}) - {best_safe['safety']}")
        
        # Patrón de juego sugerido
        if self.dispersion_patterns['tendency'] == 'disperso':
            print("\nPATRON SUGERIDO: Evitar esquinas y bordes, preferir zonas centrales")
        elif self.dispersion_patterns['tendency'] == 'agrupado':
            print("\nPATRON SUGERIDO: Evitar zonas centrales, preferir dispersion")
        else:
            print("\nPATRON SUGERIDO: Juego equilibrado entre centro y bordes")

def main():
    """Función principal mejorada"""
    predictor = EnhancedMinesPredictorV6("games/minas-2025-08-16.csv")
    predictions, safe_zones, prob_matrix = predictor.predict_next_game_enhanced()
    
    return predictor, predictions, safe_zones

if __name__ == "__main__":
    main()