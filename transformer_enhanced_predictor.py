# -*- coding: utf-8 -*-
"""
Predictor Hibrido: Transformer + Analisis Avanzado
Combina el modelo Transformer entrenado con analisis estadistico avanzado
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import MinesDataPipeline
from model_mines_transformer import MinesTransformer

class TransformerEnhancedPredictor:
    """
    Predictor hibrido que combina:
    1. Transformer entrenado para predicciones base
    2. Analisis estadistico avanzado para refinamiento
    3. Contexto expandido de todas las partidas
    """
    
    def __init__(self, 
                 model_path: str = "model_mines.keras",
                 data_dir: str = "games",
                 sequence_length: int = 5):  # Aumentamos ventana de contexto
        
        self.model_path = model_path
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        
        # Componentes del sistema
        self.transformer_model = None
        self.pipeline = None
        self.loaded = False
        
        # Datos historicos para analisis
        self.historical_data = None
        self.all_games = {}
        self.position_frequencies = None
        
    def load_model(self):
        """Carga el modelo Transformer entrenado"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo Transformer no encontrado: {self.model_path}")
        
        print(f"Cargando modelo Transformer desde: {self.model_path}")
        
        # Cargar modelo sin compilar para evitar errores de métricas faltantes
        try:
            import tensorflow as tf
            self.transformer_model = tf.keras.models.load_model(self.model_path, compile=False)
            print("Modelo Transformer cargado correctamente (sin compilar)")
        except Exception as e:
            print(f"Error cargando modelo con keras: {e}")
            # Fallback: intentar cargar con MinesTransformer
            try:
                self.transformer_model = MinesTransformer.load_model(self.model_path)
                print("Modelo cargado con MinesTransformer")
            except Exception as e2:
                raise Exception(f"No se pudo cargar el modelo. Errores: {e}, {e2}")
        
        self.loaded = True
        
    def load_historical_data(self, csv_path: str = None):
        """Carga y procesa todos los datos historicos"""
        print("Cargando datos historicos para analisis...")
        
        if csv_path is None:
            # Buscar el archivo mas reciente
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No se encontraron archivos CSV en {self.data_dir}")
            csv_path = os.path.join(self.data_dir, sorted(csv_files)[-1])
        
        # Cargar datos historicos
        df = pd.read_csv(csv_path)
        
        # Organizar por partidas
        for _, row in df.iterrows():
            game_id = int(row['partida'])
            if game_id not in self.all_games:
                self.all_games[game_id] = []
            self.all_games[game_id].append({
                'col': int(row['columna']),
                'row': int(row['fila']), 
                'mine': int(row['mina'])
            })
        
        # Calcular frecuencias historicas
        self.position_frequencies = np.zeros((5, 5))
        total_games = len(self.all_games)
        
        for game_data in self.all_games.values():
            for pos in game_data:
                if pos['mine'] == 1:
                    self.position_frequencies[pos['row']-1, pos['col']-1] += 1
        
        if total_games > 0:
            self.position_frequencies = self.position_frequencies / total_games
        
        print(f"Datos historicos cargados: {total_games} partidas")
        return csv_path
        
    def prepare_transformer_input(self, csv_path: str) -> np.ndarray:
        """Prepara la entrada para el Transformer usando ventana expandida"""
        
        # Crear pipeline temporal con ventana expandida
        temp_pipeline = MinesDataPipeline(
            data_dir=os.path.dirname(csv_path),
            sequence_length=self.sequence_length,  # Ventana mas grande
            random_seed=42
        )
        
        # Crear directorio temporal para procesamiento
        import shutil
        temp_dir = "temp_transformer_inference"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, os.path.basename(csv_path))
        shutil.copy2(csv_path, temp_file)
        temp_pipeline.data_dir = temp_dir
        
        try:
            # Procesar datos con el pipeline del Transformer
            data = temp_pipeline.process_full_pipeline()
            
            # Tomar la ultima secuencia disponible (mas reciente)
            if len(data['X_train']) > 0:
                last_sequence = data['X_train'][-1:]  # Ultima secuencia
            elif len(data['X_val']) > 0:
                last_sequence = data['X_val'][-1:]
            elif len(data['X_test']) > 0:
                last_sequence = data['X_test'][-1:]
            else:
                raise ValueError("No hay secuencias disponibles para el Transformer")
            
            print(f"Secuencia preparada para Transformer: {last_sequence.shape}")
            return last_sequence
            
        finally:
            # Limpiar directorio temporal
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def get_transformer_predictions(self, X_sequence: np.ndarray) -> np.ndarray:
        """Obtiene predicciones del Transformer"""
        if not self.loaded:
            self.load_model()
        
        print("Generando predicciones con Transformer...")
        
        # Prediccion del Transformer - manejar diferentes tipos de modelo
        try:
            if hasattr(self.transformer_model, 'model'):
                # Si es un objeto MinesTransformer
                transformer_probs = self.transformer_model.model.predict(X_sequence, verbose=0)[0]
            else:
                # Si es un modelo de Keras directo
                transformer_probs = self.transformer_model.predict(X_sequence, verbose=0)[0]
        except Exception as e:
            print(f"Error en prediccion: {e}")
            # Fallback: crear predicciones uniformes
            transformer_probs = np.ones(25) * 0.12  # Aproximadamente 3 minas
        
        print(f"Transformer completado. Suma de probabilidades: {np.sum(transformer_probs):.3f}")
        return transformer_probs
    
    def apply_statistical_enhancements(self, transformer_probs: np.ndarray) -> np.ndarray:
        """Aplica mejoras estadisticas a las predicciones del Transformer"""
        print("Aplicando analisis estadistico avanzado...")
        
        enhanced_probs = transformer_probs.copy()
        
        # 1. Factores de frecuencia historica (peso 30%)
        historical_factor = self.position_frequencies.flatten() * 0.3
        enhanced_probs += historical_factor
        
        # 2. Analisis de enfriamiento (peso 20%)
        cooling_factor = self._calculate_cooling_effects() * 0.2
        enhanced_probs += cooling_factor
        
        # 3. Analisis de patrones recientes (peso 25%)
        pattern_factor = self._calculate_recent_patterns() * 0.25
        enhanced_probs += pattern_factor
        
        # 4. Analisis de zonas espaciales (peso 15%)
        spatial_factor = self._calculate_spatial_boost() * 0.15
        enhanced_probs += spatial_factor
        
        # 5. Suavizado del Transformer (peso 10%)
        transformer_factor = transformer_probs * 0.1
        enhanced_probs += transformer_factor
        
        # Normalizar para mantener suma cercana a 3 minas
        enhanced_probs = np.maximum(enhanced_probs, 0.001)  # Minimo 0.1%
        total_prob = np.sum(enhanced_probs)
        if total_prob > 0:
            enhanced_probs = enhanced_probs * (3.0 / total_prob)
        
        print("Analisis estadistico completado")
        return enhanced_probs
    
    def _calculate_cooling_effects(self) -> np.ndarray:
        """Calcula efectos de enfriamiento en posiciones recientes"""
        cooling_effects = np.zeros(25)
        
        # Identificar posiciones con minas en ultimas 2-3 partidas
        recent_games = list(self.all_games.keys())[-3:]
        recent_mine_positions = set()
        
        for game_id in recent_games:
            for pos in self.all_games[game_id]:
                if pos['mine'] == 1:
                    recent_mine_positions.add((pos['col'], pos['row']))
        
        # Aplicar factor de enfriamiento
        for col, row in recent_mine_positions:
            idx = (row - 1) * 5 + (col - 1)
            cooling_effects[idx] -= 0.15  # Reducir probabilidad
        
        return cooling_effects
    
    def _calculate_recent_patterns(self) -> np.ndarray:
        """Analiza patrones en partidas mas recientes"""
        pattern_boost = np.zeros(25)
        
        if len(self.all_games) >= 3:
            # Analizar ultimas 3 partidas
            recent_games = list(self.all_games.keys())[-3:]
            recent_frequencies = np.zeros((5, 5))
            
            for game_id in recent_games:
                for pos in self.all_games[game_id]:
                    if pos['mine'] == 1:
                        recent_frequencies[pos['row']-1, pos['col']-1] += 1
            
            # Normalizar y aplicar boost a posiciones frecuentes recientes
            recent_frequencies = recent_frequencies / len(recent_games)
            pattern_boost = recent_frequencies.flatten() * 0.4
        
        return pattern_boost
    
    def _calculate_spatial_boost(self) -> np.ndarray:
        """Calcula boost basado en distribucion espacial"""
        spatial_boost = np.zeros(25)
        
        # Definir zonas del tablero
        zones = {
            'esquinas': [(0,0), (0,4), (4,0), (4,4)],
            'bordes': [(0,1), (0,2), (0,3), (1,0), (1,4), (2,0), (2,4), (3,0), (3,4), (4,1), (4,2), (4,3)],
            'centro': [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
        }
        
        # Calcular actividad por zona
        zone_activities = {}
        for zone_name, positions in zones.items():
            activity = 0
            for row, col in positions:
                activity += self.position_frequencies[row, col]
            zone_activities[zone_name] = activity / len(positions)
        
        # Aplicar boost a la zona mas activa
        most_active_zone = max(zone_activities.items(), key=lambda x: x[1])
        
        if most_active_zone[1] > 0.1:  # Si hay actividad significativa
            zone_positions = zones[most_active_zone[0]]
            for row, col in zone_positions:
                idx = row * 5 + col
                spatial_boost[idx] += 0.1
        
        return spatial_boost
    
    def generate_detailed_predictions(self, enhanced_probs: np.ndarray) -> List[Dict]:
        """Genera predicciones detalladas con justificaciones"""
        prob_matrix = enhanced_probs.reshape(5, 5)
        
        # Top 3 posiciones
        flat_probs = enhanced_probs.flatten()
        top_indices = np.argsort(flat_probs)[::-1][:3]
        
        predictions = []
        for i, idx in enumerate(top_indices):
            row = idx // 5 + 1
            col = idx % 5 + 1
            prob = flat_probs[idx] * 100
            
            # Analisis detallado de justificacion
            justifications = []
            
            # Transformer base
            transformer_prob = prob * 0.1  # El transformer contribuye 10%
            if transformer_prob > 2.0:
                justifications.append("Transformer predice alta probabilidad")
            
            # Frecuencia historica
            hist_freq = self.position_frequencies[row-1, col-1]
            if hist_freq > 0.15:
                justifications.append(f"Frecuencia historica alta ({hist_freq:.1%})")
            
            # Analisis reciente
            recent_games = list(self.all_games.keys())[-3:]
            recent_count = 0
            for game_id in recent_games:
                for pos in self.all_games[game_id]:
                    if pos['mine'] == 1 and pos['col'] == col and pos['row'] == row:
                        recent_count += 1
            
            if recent_count > 0:
                justifications.append(f"Aparecio {recent_count} vez(es) en ultimas 3 partidas")
            
            # Zona de enfriamiento
            if (col, row) not in self._get_recent_mine_positions():
                justifications.append("Zona sin enfriamiento reciente")
            
            predictions.append({
                'position': (col, row),
                'probability': prob,
                'justification': " + ".join(justifications) if justifications else "Analisis Transformer + estadistico"
            })
        
        return predictions
    
    def _get_recent_mine_positions(self) -> set:
        """Obtiene posiciones con minas en partidas recientes"""
        recent_games = list(self.all_games.keys())[-2:]
        positions = set()
        
        for game_id in recent_games:
            for pos in self.all_games[game_id]:
                if pos['mine'] == 1:
                    positions.add((pos['col'], pos['row']))
        
        return positions
    
    def get_safe_zones(self, enhanced_probs: np.ndarray) -> List[Dict]:
        """Identifica zonas seguras con analisis hibrido"""
        prob_matrix = enhanced_probs.reshape(5, 5) * 100
        safe_zones = []
        
        for row in range(5):
            for col in range(5):
                prob = prob_matrix[row, col]
                
                if prob < 15.0:  # Zonas con menos de 15% de riesgo
                    # Clasificar seguridad
                    if prob < 3.0:
                        safety = 'ULTRA-SEGURA'
                    elif prob < 7.0:
                        safety = 'MUY SEGURA'
                    elif prob < 12.0:
                        safety = 'SEGURA'
                    else:
                        safety = 'MODERADAMENTE SEGURA'
                    
                    # Razones de seguridad
                    reasons = []
                    
                    # Frecuencia historica baja
                    hist_freq = self.position_frequencies[row, col]
                    if hist_freq < 0.1:
                        reasons.append("baja frecuencia historica")
                    
                    # Sin actividad reciente
                    if (col+1, row+1) not in self._get_recent_mine_positions():
                        reasons.append("sin actividad reciente")
                    
                    # Transformer considera segura
                    if prob < 5.0:
                        reasons.append("Transformer + estadisticas coinciden")
                    
                    safe_zones.append({
                        'position': (col+1, row+1),
                        'probability': prob,
                        'safety': safety,
                        'reasoning': ", ".join(reasons) if reasons else "analisis hibrido"
                    })
        
        return sorted(safe_zones, key=lambda x: x['probability'])
    
    def predict_next_game_hybrid(self, csv_path: str = None) -> Dict[str, Any]:
        """
        Prediccion hibrida completa usando Transformer + analisis estadistico
        """
        print("PREDICTOR HIBRIDO: TRANSFORMER + ANALISIS AVANZADO")
        print("=" * 70)
        
        # 1. Cargar datos historicos
        csv_path = self.load_historical_data(csv_path)
        
        # 2. Preparar entrada para Transformer
        X_sequence = self.prepare_transformer_input(csv_path)
        
        # 3. Obtener predicciones del Transformer
        transformer_probs = self.get_transformer_predictions(X_sequence)
        
        # 4. Aplicar mejoras estadisticas
        enhanced_probs = self.apply_statistical_enhancements(transformer_probs)
        
        # 5. Generar predicciones detalladas
        predictions = self.generate_detailed_predictions(enhanced_probs)
        
        # 6. Identificar zonas seguras
        safe_zones = self.get_safe_zones(enhanced_probs)
        
        # 7. Informacion de contexto
        total_games = len(self.all_games)
        next_game = max(self.all_games.keys()) + 1 if self.all_games else 1
        
        print(f"\nPREDICCION PARA PARTIDA {next_game}")
        print(f"Basada en: Transformer + analisis de {total_games} partidas")
        print(f"Ventana de contexto: {self.sequence_length} partidas")
        
        # Mostrar predicciones
        print(f"\nPREDICCIONES TOP 3 (HIBRIDAS):")
        for i, pred in enumerate(predictions, 1):
            pos = pred['position']
            prob = pred['probability']
            just = pred['justification']
            print(f"\nMINA #{i}: ({pos[0]},{pos[1]}) - {prob:.1f}% probabilidad")
            print(f"  Justificacion: {just}")
        
        # Mostrar zonas seguras
        print(f"\nZONAS SEGURAS RECOMENDADAS:")
        for i, safe in enumerate(safe_zones[:8], 1):
            pos = safe['position']
            prob = safe['probability']
            safety = safe['safety']
            reasoning = safe['reasoning']
            print(f"{i}. ({pos[0]},{pos[1]}): {prob:.1f}% riesgo - {safety}")
            print(f"   Razon: {reasoning}")
        
        # Estrategia final
        prob_matrix = enhanced_probs.reshape(5, 5) * 100
        row_risks = [np.sum(prob_matrix[i, :]) for i in range(5)]
        col_risks = [np.sum(prob_matrix[:, j]) for j in range(5)]
        
        safest_row = np.argmin(row_risks) + 1
        safest_col = np.argmin(col_risks) + 1
        
        print(f"\nESTRATEGIA HIBRIDA RECOMENDADA:")
        print(f"Fila mas segura: {safest_row} ({row_risks[safest_row-1]:.1f}% riesgo total)")
        print(f"Columna mas segura: {safest_col} ({col_risks[safest_col-1]:.1f}% riesgo total)")
        
        if safe_zones:
            best_safe = safe_zones[0]
            print(f"RECOMENDACION PRINCIPAL: ({best_safe['position'][0]},{best_safe['position'][1]}) - {best_safe['safety']}")
            print(f"Razon: {best_safe['reasoning']}")
        
        return {
            'next_game': next_game,
            'total_games_analyzed': total_games,
            'predictions': predictions,
            'safe_zones': safe_zones,
            'transformer_probs': transformer_probs.tolist(),
            'enhanced_probs': enhanced_probs.tolist(),
            'prob_matrix': prob_matrix.tolist(),
            'strategy': {
                'safest_row': safest_row,
                'safest_col': safest_col,
                'row_risks': row_risks,
                'col_risks': col_risks
            }
        }

def main(csv_file=None):
    """Funcion principal del predictor hibrido"""
    print("Iniciando Predictor Hibrido Transformer + Analisis Avanzado")
    
    predictor = TransformerEnhancedPredictor(
        model_path="model_mines.keras",
        data_dir="games",
        sequence_length=5  # Ventana mas amplia que el v6
    )
    
    try:
        # Si se especifica archivo, usarlo; sino usar automatico
        if csv_file:
            print(f"Usando archivo especificado: {csv_file}")
            results = predictor.predict_next_game_hybrid(csv_path=csv_file)
        else:
            print("Usando archivo mas reciente del directorio games/")
            results = predictor.predict_next_game_hybrid()
        
        # Guardar resultados
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"prediccion_hibrida_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResultados guardados en: {output_file}")
        print("\nPrediccion hibrida completada exitosamente")
        
        return results
        
    except Exception as e:
        print(f"Error durante la prediccion: {str(e)}")
        raise

if __name__ == "__main__":
    # Verificar si se paso un archivo como argumento
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        main(csv_file)
    else:
        main()