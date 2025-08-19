import pandas as pd
import numpy as np
from collections import defaultdict

class FutureGamesPredictor:
    """Predictor optimizado para futuros juegos basado en últimas 3 partidas"""
    
    def __init__(self, csv_file_path):
        self.csv_file = csv_file_path
        self.load_data()
        
    def load_data(self):
        """Carga datos y obtiene últimas 3 partidas"""
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
        
        # Obtener últimas 3 partidas completadas
        sorted_games = sorted(self.all_games.keys())
        self.last_3_games = sorted_games[-3:] if len(sorted_games) >= 3 else sorted_games
        self.next_game_number = max(sorted_games) + 1 if sorted_games else 1
        
    def analyze_last_3_patterns(self):
        """Analiza patrones específicos de las últimas 3 partidas"""
        patterns = {
            'mine_positions': [],
            'position_count': defaultdict(int),
            'row_count': defaultdict(int),
            'col_count': defaultdict(int),
            'geometry_types': []
        }
        
        print(f"Analizando últimas 3 partidas: {self.last_3_games}")
        
        for game_id in self.last_3_games:
            game_mines = []
            print(f"\nPartida {game_id}:")
            
            for pos in self.all_games[game_id]:
                if pos['mine'] == 1:
                    mine_pos = (pos['col'], pos['row'])
                    game_mines.append(mine_pos)
                    patterns['mine_positions'].append(mine_pos)
                    patterns['position_count'][mine_pos] += 1
                    patterns['row_count'][pos['row']] += 1
                    patterns['col_count'][pos['col']] += 1
                    print(f"  Mina en ({pos['col']},{pos['row']})")
            
            # Analizar geometría del juego
            geometry = self.analyze_game_geometry(game_mines)
            patterns['geometry_types'].append(geometry)
            print(f"  Geometría: {geometry}")
        
        return patterns
    
    def analyze_game_geometry(self, mines):
        """Analiza la geometría de distribución de las 3 minas"""
        if len(mines) != 3:
            return "incompleto"
        
        # Calcular distancias entre minas
        distances = []
        for i in range(3):
            for j in range(i+1, 3):
                dist = abs(mines[i][0] - mines[j][0]) + abs(mines[i][1] - mines[j][1])
                distances.append(dist)
        
        avg_distance = sum(distances) / len(distances)
        
        # Clasificar por dispersión
        if avg_distance > 5:
            return "muy_disperso"
        elif avg_distance > 3:
            return "disperso"
        elif avg_distance > 2:
            return "moderado"
        else:
            return "agrupado"
    
    def calculate_adaptive_probabilities(self, patterns):
        """Calcula probabilidades adaptándose a las últimas 3 partidas"""
        prob_matrix = np.ones((5, 5)) * 0.1  # Base mínima
        
        # Factor 1: Frecuencia directa en últimas 3 partidas (40% peso)
        total_mines = len(patterns['mine_positions'])
        for (col, row), count in patterns['position_count'].items():
            weight = count / total_mines * 0.4
            prob_matrix[row-1, col-1] += weight
        
        # Factor 2: Frecuencia por filas/columnas (30% peso)
        for row, count in patterns['row_count'].items():
            row_weight = count / total_mines * 0.3
            prob_matrix[row-1, :] += row_weight
        
        for col, count in patterns['col_count'].items():
            col_weight = count / total_mines * 0.3
            prob_matrix[:, col-1] += col_weight
        
        # Factor 3: Adaptación geométrica (30% peso)
        dominant_geometry = max(set(patterns['geometry_types']), key=patterns['geometry_types'].count)
        self.apply_geometry_adaptation(prob_matrix, dominant_geometry)
        
        # Factor 4: Anti-repetición (reducir posiciones muy recientes)
        for (col, row), count in patterns['position_count'].items():
            if count >= 2:  # Apareció 2+ veces
                prob_matrix[row-1, col-1] *= 0.7  # Reducir 30%
        
        # Normalizar para ~3 minas totales
        prob_matrix = prob_matrix / np.sum(prob_matrix) * 3.0
        prob_matrix = np.maximum(prob_matrix, 0.005)  # Mínimo 0.5%
        
        return prob_matrix
    
    def apply_geometry_adaptation(self, prob_matrix, geometry_type):
        """Aplica adaptación según tipo de geometría dominante"""
        if geometry_type == "muy_disperso":
            # Potenciar esquinas y bordes
            prob_matrix[0, :] += 0.05  # fila 1
            prob_matrix[4, :] += 0.05  # fila 5
            prob_matrix[:, 0] += 0.05  # columna 1
            prob_matrix[:, 4] += 0.05  # columna 5
            prob_matrix[2, 2] -= 0.03  # reducir centro
            
        elif geometry_type == "agrupado":
            # Potenciar zonas centrales y adyacentes
            prob_matrix[1:4, 1:4] += 0.04
            prob_matrix[2, 2] += 0.02  # boost centro extra
            
        else:  # disperso o moderado
            # Equilibrio entre bordes y centro
            prob_matrix[[0, 4], :] += 0.02
            prob_matrix[:, [0, 4]] += 0.02
            prob_matrix[1:4, 1:4] += 0.01
    
    def get_top_predictions(self, prob_matrix, patterns):
        """Obtiene las top 3 predicciones con justificación"""
        flat_probs = prob_matrix.flatten()
        top_indices = np.argsort(flat_probs)[::-1][:3]
        
        predictions = []
        for i, idx in enumerate(top_indices):
            row = idx // 5 + 1
            col = idx % 5 + 1
            prob = flat_probs[idx] * 100
            
            # Crear justificación
            justification = []
            if (col, row) in patterns['position_count']:
                count = patterns['position_count'][(col, row)]
                justification.append(f"Apareció {count} veces en últimas 3 partidas")
            
            if patterns['row_count'][row] > 1:
                justification.append(f"Fila {row} activa recientemente")
            
            if patterns['col_count'][col] > 1:
                justification.append(f"Columna {col} activa recientemente")
            
            if row in [1, 5] or col in [1, 5]:
                justification.append("Posición de borde")
            
            predictions.append({
                'position': (col, row),
                'probability': prob,
                'justification': " + ".join(justification) if justification else "Patrón estadístico adaptativo"
            })
        
        return predictions
    
    def analyze_safe_rows_cols(self, prob_matrix):
        """Analiza seguridad por filas y columnas completas"""
        row_risks = []
        col_risks = []
        
        # Análisis por filas
        for row in range(5):
            row_prob = np.mean(prob_matrix[row, :]) * 100
            row_risks.append({
                'row': row + 1,
                'avg_risk': row_prob,
                'safest_cell': np.argmin(prob_matrix[row, :]) + 1,
                'safest_risk': np.min(prob_matrix[row, :]) * 100,
                'cells_under_5pct': np.sum(prob_matrix[row, :] < 0.05)
            })
        
        # Análisis por columnas
        for col in range(5):
            col_prob = np.mean(prob_matrix[:, col]) * 100
            col_risks.append({
                'col': col + 1,
                'avg_risk': col_prob,
                'safest_cell': np.argmin(prob_matrix[:, col]) + 1,
                'safest_risk': np.min(prob_matrix[:, col]) * 100,
                'cells_under_5pct': np.sum(prob_matrix[:, col] < 0.05)
            })
        
        return sorted(row_risks, key=lambda x: x['avg_risk']), sorted(col_risks, key=lambda x: x['avg_risk'])
    
    def get_safe_zones(self, prob_matrix):
        """Identifica zonas más seguras con análisis detallado"""
        safe_positions = []
        
        for row in range(5):
            for col in range(5):
                prob = prob_matrix[row, col] * 100
                
                # Clasificar por nivel de seguridad
                if prob < 2.0:
                    safety = 'ULTRA-SEGURA'
                elif prob < 4.0:
                    safety = 'MUY SEGURA'
                elif prob < 6.0:
                    safety = 'SEGURA'
                elif prob < 8.0:
                    safety = 'MODERADA'
                else:
                    continue  # Solo mostrar las relativamente seguras
                
                # Agregar contexto posicional
                position_type = []
                if row == 0 or row == 4:
                    position_type.append('borde')
                if col == 0 or col == 4:
                    position_type.append('lateral')
                if row == 2 and col == 2:
                    position_type.append('centro')
                if (row == 0 and col == 0) or (row == 0 and col == 4) or (row == 4 and col == 0) or (row == 4 and col == 4):
                    position_type.append('esquina')
                
                safe_positions.append({
                    'position': (col+1, row+1),
                    'probability': prob,
                    'safety': safety,
                    'position_type': position_type
                })
        
        return sorted(safe_positions, key=lambda x: x['probability'])
    
    def analyze_strategic_zones(self, prob_matrix):
        """Analiza zonas estratégicas para diferentes estilos de juego"""
        strategies = {
            'conservador': [],
            'balanceado': [],
            'agresivo': []
        }
        
        for row in range(5):
            for col in range(5):
                prob = prob_matrix[row, col] * 100
                pos = (col+1, row+1)
                
                if prob < 3.0:
                    strategies['conservador'].append({'pos': pos, 'risk': prob})
                if prob < 6.0:
                    strategies['balanceado'].append({'pos': pos, 'risk': prob})
                if prob < 10.0:
                    strategies['agresivo'].append({'pos': pos, 'risk': prob})
        
        return strategies
    
    def predict_next_game(self):
        """Predice la siguiente partida basándose en las últimas 3"""
        print("PREDICTOR PARA FUTUROS JUEGOS - BASADO EN ULTIMAS 3 PARTIDAS")
        print("=" * 70)
        
        print(f"\nPREDICCION PARA PARTIDA {self.next_game_number}")
        
        # Analizar patrones de las últimas 3 partidas
        patterns = self.analyze_last_3_patterns()
        
        # Mostrar resumen de patrones
        print(f"\nRESUMEN DE PATRONES (últimas 3 partidas):")
        print(f"- Total minas analizadas: {len(patterns['mine_positions'])}")
        print(f"- Posiciones más frecuentes: {dict(sorted(patterns['position_count'].items(), key=lambda x: x[1], reverse=True)[:3])}")
        print(f"- Filas más activas: {dict(sorted(patterns['row_count'].items(), key=lambda x: x[1], reverse=True)[:3])}")
        print(f"- Columnas más activas: {dict(sorted(patterns['col_count'].items(), key=lambda x: x[1], reverse=True)[:3])}")
        print(f"- Tipos de geometría: {patterns['geometry_types']}")
        
        # Calcular probabilidades adaptativas
        prob_matrix = self.calculate_adaptive_probabilities(patterns)
        
        # Obtener predicciones
        predictions = self.get_top_predictions(prob_matrix, patterns)
        
        print(f"\nPREDICCIONES TOP 3 PARA PARTIDA {self.next_game_number}:")
        for i, pred in enumerate(predictions, 1):
            pos = pred['position']
            prob = pred['probability']
            just = pred['justification']
            print(f"\nMINA #{i}: ({pos[0]},{pos[1]}) - {prob:.1f}% probabilidad")
            print(f"  Justificación: {just}")
        
        # Zonas seguras
        safe_zones = self.get_safe_zones(prob_matrix)
        
        print(f"\nZONAS MAS SEGURAS PARA JUGAR:")
        for i, safe in enumerate(safe_zones[:5], 1):
            pos = safe['position']
            prob = safe['probability']
            safety = safe['safety']
            print(f"{i}. ({pos[0]},{pos[1]}): {prob:.1f}% riesgo - {safety}")
        
        # Análisis por filas y columnas
        safe_rows, safe_cols = self.analyze_safe_rows_cols(prob_matrix)
        
        print(f"\n=== ANÁLISIS POR FILAS ===")
        print("Filas ordenadas por seguridad (menor riesgo = más segura):")
        for i, row_info in enumerate(safe_rows[:3], 1):
            print(f"{i}. FILA {row_info['row']}: {row_info['avg_risk']:.1f}% riesgo promedio")
            print(f"   └─ Casilla más segura: ({row_info['safest_cell']},{row_info['row']}) con {row_info['safest_risk']:.1f}% riesgo")
            print(f"   └─ Casillas bajo 5% riesgo: {row_info['cells_under_5pct']}/5")
        
        print(f"\n=== ANÁLISIS POR COLUMNAS ===")
        print("Columnas ordenadas por seguridad (menor riesgo = más segura):")
        for i, col_info in enumerate(safe_cols[:3], 1):
            print(f"{i}. COLUMNA {col_info['col']}: {col_info['avg_risk']:.1f}% riesgo promedio")
            print(f"   └─ Casilla más segura: ({col_info['col']},{col_info['safest_cell']}) con {col_info['safest_risk']:.1f}% riesgo")
            print(f"   └─ Casillas bajo 5% riesgo: {col_info['cells_under_5pct']}/5")
        
        # Zonas seguras expandidas
        print(f"\n=== CASILLAS SEGURAS RECOMENDADAS ===")
        for i, safe in enumerate(safe_zones[:8], 1):
            pos = safe['position']
            prob = safe['probability']
            safety = safe['safety']
            pos_type = ", ".join(safe['position_type']) if safe['position_type'] else "interior"
            print(f"{i}. ({pos[0]},{pos[1]}): {prob:.1f}% riesgo - {safety} ({pos_type})")
        
        # Estrategias por estilo de juego
        strategies = self.analyze_strategic_zones(prob_matrix)
        
        print(f"\n=== ESTRATEGIAS RECOMENDADAS ===")
        print(f"\n🛡️  CONSERVADOR (máximo 3% riesgo):")
        if strategies['conservador']:
            for i, move in enumerate(strategies['conservador'][:5], 1):
                print(f"   {i}. ({move['pos'][0]},{move['pos'][1]}) - {move['risk']:.1f}% riesgo")
        else:
            print("   ⚠️  No hay casillas ultra-seguras disponibles")
        
        print(f"\n⚖️  BALANCEADO (máximo 6% riesgo):")
        for i, move in enumerate(strategies['balanceado'][:6], 1):
            print(f"   {i}. ({move['pos'][0]},{move['pos'][1]}) - {move['risk']:.1f}% riesgo")
        
        print(f"\n⚔️  AGRESIVO (máximo 10% riesgo):")
        for i, move in enumerate(strategies['agresivo'][:8], 1):
            print(f"   {i}. ({move['pos'][0]},{move['pos'][1]}) - {move['risk']:.1f}% riesgo")
        
        print(f"\n=== POSICIONES DE ALTO RIESGO (EVITAR) ===")
        for pred in predictions:
            pos = pred['position']
            prob = pred['probability']
            print(f"❌ ({pos[0]},{pos[1]}) - {prob:.1f}% probabilidad de mina")
        
        # Análisis adicional de tendencias
        print(f"\n=== ANÁLISIS DE TENDENCIAS ==")
        self.analyze_advanced_patterns(patterns, prob_matrix)
        
        print(f"\n{'='*70}")
        print(f"📊 RESUMEN: {len(safe_zones)} casillas seguras detectadas")
        print(f"🎯 RECOMENDACIÓN: Comenzar con la fila {safe_rows[0]['row']} o columna {safe_cols[0]['col']}")
        print(f"⚡ SIGUIENTE ACTUALIZACIÓN: Después de completar partida {self.next_game_number}")
        print(f"{'='*70}")
        
        return predictions, safe_zones
    
    def analyze_advanced_patterns(self, patterns, prob_matrix):
        """Análisis avanzado de patrones y tendencias"""
        
        # Análisis de distribución temporal
        print(f"📈 DISTRIBUCIÓN TEMPORAL:")
        recent_positions = patterns['mine_positions'][-6:]  # Últimas 6 minas
        early_positions = patterns['mine_positions'][:3]   # Primeras 3 minas
        
        print(f"   • Tendencia hacia bordes: {self.calculate_border_tendency(recent_positions):.1f}%")
        print(f"   • Cambio en dispersión: {self.analyze_dispersion_change(patterns)}")
        
        # Análisis de zonas evitadas
        avoided_zones = self.find_avoided_zones(prob_matrix)
        if avoided_zones:
            print(f"🚫 ZONAS SISTEMÁTICAMENTE EVITADAS:")
            for zone in avoided_zones[:3]:
                print(f"   • Zona {zone['area']}: {zone['avoidance']:.1f}% menos minas que esperado")
        
        # Análisis de correlaciones
        print(f"🔗 CORRELACIONES DETECTADAS:")
        correlations = self.analyze_correlations(patterns)
        for corr in correlations[:2]:
            print(f"   • {corr}")
    
    def calculate_border_tendency(self, positions):
        """Calcula tendencia hacia posiciones de borde"""
        if not positions:
            return 0
        border_count = sum(1 for pos in positions if pos[0] in [1,5] or pos[1] in [1,5])
        return (border_count / len(positions)) * 100
    
    def analyze_dispersion_change(self, patterns):
        """Analiza cambio en la dispersión de las minas"""
        geometries = patterns['geometry_types']
        if len(geometries) < 2:
            return "Insuficientes datos"
        
        dispersion_values = {'agrupado': 1, 'moderado': 2, 'disperso': 3, 'muy_disperso': 4}
        recent_avg = sum(dispersion_values.get(g, 2) for g in geometries[-2:]) / 2
        early_avg = sum(dispersion_values.get(g, 2) for g in geometries[:-2]) / max(1, len(geometries)-2)
        
        if recent_avg > early_avg + 0.5:
            return "Incrementando dispersión"
        elif recent_avg < early_avg - 0.5:
            return "Reduciendo dispersión"
        else:
            return "Dispersión estable"
    
    def find_avoided_zones(self, prob_matrix):
        """Identifica zonas sistemáticamente evitadas"""
        expected_prob = 1.0 / 25 * 3  # Probabilidad esperada por casilla
        avoided = []
        
        zones = {
            'centro (2-4, 2-4)': prob_matrix[1:4, 1:4],
            'bordes superiores': prob_matrix[0, :],
            'bordes inferiores': prob_matrix[4, :],
            'bordes izquierdos': prob_matrix[:, 0],
            'bordes derechos': prob_matrix[:, 4]
        }
        
        for zone_name, zone_data in zones.items():
            avg_prob = np.mean(zone_data)
            if avg_prob < expected_prob * 0.6:  # 40% menos que esperado
                avoidance = ((expected_prob - avg_prob) / expected_prob) * 100
                avoided.append({'area': zone_name, 'avoidance': avoidance})
        
        return sorted(avoided, key=lambda x: x['avoidance'], reverse=True)
    
    def analyze_correlations(self, patterns):
        """Analiza correlaciones en los patrones"""
        correlations = []
        
        # Correlación fila-columna
        row_counts = patterns['row_count']
        col_counts = patterns['col_count']
        
        max_row = max(row_counts.keys()) if row_counts else 0
        max_col = max(col_counts.keys()) if col_counts else 0
        
        if max_row > 0 and max_col > 0:
            if row_counts.get(max_row, 0) > 1 and col_counts.get(max_col, 0) > 1:
                correlations.append(f"Fila {max_row} y Columna {max_col} muestran alta actividad correlacionada")
        
        # Patrón de repetición
        position_counts = patterns['position_count']
        repeated_positions = [pos for pos, count in position_counts.items() if count > 1]
        if repeated_positions:
            correlations.append(f"Posiciones con repetición detectada: {repeated_positions}")
        
        if not correlations:
            correlations.append("No se detectaron correlaciones significativas")
        
        return correlations

def main():
    """Función principal para uso futuro"""
    predictor = FutureGamesPredictor("games/minas-2025-08-16.csv")
    predictor.predict_next_game()

if __name__ == "__main__":
    main()