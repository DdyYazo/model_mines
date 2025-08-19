# -*- coding: utf-8 -*-
"""
Script de Inferencia para Modelo Transformer de Minas
Función para predecir la siguiente partida y análisis de seguridad
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import MinesDataPipeline
from model_mines_transformer import MinesTransformer

class MinesPredictor:
    """Predictor de minas usando modelo Transformer entrenado"""
    
    def __init__(self, model_path: str = "model_mines.keras"):
        self.model_path = model_path
        self.model = None
        self.pipeline = None
        self.loaded = False
        
    def load_model(self):
        """Carga el modelo entrenado"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {self.model_path}")
        
        print(f" Cargando modelo desde: {self.model_path}")
        self.model = MinesTransformer.load_model(self.model_path)
        self.loaded = True
        print(" Modelo cargado correctamente")
        
    def _prepare_sequence_data(self, csv_path: str, sequence_length: int = 3) -> np.ndarray:
        """Prepara los datos de secuencia para inferencia"""
        
        # Crear pipeline temporal para procesar datos
        temp_pipeline = MinesDataPipeline(
            data_dir=os.path.dirname(csv_path),
            sequence_length=sequence_length,
            random_seed=42
        )
        
        # Procesar solo el archivo específico o todos los disponibles
        if os.path.isfile(csv_path):
            # Copiar archivo a directorio temporal para procesamiento
            import shutil
            temp_dir = "temp_inference"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, os.path.basename(csv_path))
            shutil.copy2(csv_path, temp_file)
            temp_pipeline.data_dir = temp_dir
        
        # Procesar datos
        try:
            data = temp_pipeline.process_full_pipeline()
            
            # Tomar la última secuencia disponible
            if len(data['X_train']) > 0:
                last_sequence = data['X_train'][-1:] # Última secuencia de entrenamiento
            elif len(data['X_val']) > 0:
                last_sequence = data['X_val'][-1:]   # O de validación
            elif len(data['X_test']) > 0:
                last_sequence = data['X_test'][-1:]  # O de test
            else:
                raise ValueError("No hay secuencias disponibles para inferencia")
            
            return last_sequence
            
        finally:
            # Limpiar directorio temporal si se creó
            if os.path.exists("temp_inference"):
                import shutil
                shutil.rmtree("temp_inference")
    
    def predict_next_game(self, 
                         csv_path: str = None, 
                         M: int = 4,  # Cambiado de 3 a 4
                         return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Predice las coordenadas de las M minas de la siguiente partida
        
        Args:
            csv_path: Ruta al archivo CSV con historial (None para usar directorio games/)
            M: Número de minas a predecir (por defecto 4)
            return_probabilities: Si retornar matriz completa de probabilidades
            
        Returns:
            Dict con predicciones, probabilidades y análisis de seguridad
        """
        
        if not self.loaded:
            self.load_model()
        
        # Determinar ruta de datos
        if csv_path is None:
            # Usar el archivo más reciente del directorio games
            games_dir = "games"
            csv_files = [f for f in os.listdir(games_dir) if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No se encontraron archivos CSV en {games_dir}")
            csv_path = os.path.join(games_dir, sorted(csv_files)[-1])
            print(f" Usando archivo más reciente: {csv_path}")
        
        print(f" Prediciendo siguiente partida basada en: {os.path.basename(csv_path)}")
        
        # Preparar datos de secuencia
        X_sequence = self._prepare_sequence_data(csv_path)
        print(f" Secuencia preparada: {X_sequence.shape}")
        
        # Realizar predicción
        probabilities = self.model.model.predict(X_sequence, verbose=0)[0]  # Primera (y única) muestra
        
        # Seleccionar top-M posiciones
        top_M_indices = np.argsort(probabilities)[-M:][::-1]  # Ordenar descendente
        
        # Convertir índices a coordenadas (fila, columna)
        top_M_coords = [(idx // 5 + 1, idx % 5 + 1) for idx in top_M_indices]
        
        # Crear matriz de probabilidades 5x5 para visualización
        prob_matrix = probabilities.reshape(5, 5)
        
        # Análisis de seguridad por filas y columnas
        safe_analysis = self._analyze_safety(probabilities, prob_matrix, M)
        
        # Resultado final
        result = {
            "top_m_cells": top_M_coords,
            "top_m_probabilities": [float(probabilities[idx]) for idx in top_M_indices],
            "prediction_confidence": float(np.mean([probabilities[idx] for idx in top_M_indices])),
            "total_probability_sum": float(np.sum(probabilities)),
            "safe_rows": safe_analysis["safe_rows"],
            "safe_cols": safe_analysis["safe_cols"], 
            "row_risk_scores": safe_analysis["row_risks"],
            "col_risk_scores": safe_analysis["col_risks"]
        }
        
        if return_probabilities:
            result["prob_matrix"] = prob_matrix.tolist()
            result["prob_vector"] = probabilities.tolist()
        
        return result
    
    def _analyze_safety(self, probabilities: np.ndarray, prob_matrix: np.ndarray, M: int) -> Dict[str, Any]:
        """Analiza seguridad por filas y columnas"""
        
        # Calcular probabilidades agregadas por fila y columna
        row_probs = np.sum(prob_matrix, axis=1)  # Suma por fila
        col_probs = np.sum(prob_matrix, axis=0)  # Suma por columna
        
        # Determinar filas y columnas "seguras" (baja probabilidad de contener minas)
        # Usar umbral dinámico basado en la distribución
        row_threshold = np.mean(row_probs) - 0.5 * np.std(row_probs)
        col_threshold = np.mean(col_probs) - 0.5 * np.std(col_probs)
        
        safe_rows = [i+1 for i, prob in enumerate(row_probs) if prob < row_threshold]
        safe_cols = [i+1 for i, prob in enumerate(col_probs) if prob < col_threshold]
        
        return {
            "safe_rows": safe_rows,
            "safe_cols": safe_cols,
            "row_risks": row_probs.tolist(),
            "col_risks": col_probs.tolist(),
            "row_threshold": float(row_threshold),
            "col_threshold": float(col_threshold)
        }
    
    def predict_batch(self, csv_paths: List[str], M: int = 3) -> List[Dict[str, Any]]:
        """Predice múltiples partidas en lote"""
        results = []
        
        for csv_path in csv_paths:
            try:
                result = self.predict_next_game(csv_path, M, return_probabilities=False)
                result["source_file"] = os.path.basename(csv_path)
                results.append(result)
            except Exception as e:
                print(f" Error procesando {csv_path}: {str(e)}")
                
        return results
    
    def visualize_prediction(self, prediction_result: Dict[str, Any], save_path: str = None):
        """Visualiza la predicción como mapa de calor"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        prob_matrix = np.array(prediction_result["prob_matrix"])
        top_coords = prediction_result["top_m_cells"]
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mapa de calor de probabilidades
        sns.heatmap(prob_matrix, annot=True, fmt='.3f', cmap='Reds', 
                   square=True, cbar_kws={'label': 'Probabilidad'}, ax=ax1)
        ax1.set_title('Probabilidades de Minas por Celda')
        ax1.set_xlabel('Columna')
        ax1.set_ylabel('Fila')
        
        # Marcar top-M predicciones
        for i, (fila, col) in enumerate(top_coords):
            ax1.add_patch(plt.Rectangle((col-1, fila-1), 1, 1, 
                                      fill=False, edgecolor='blue', lw=3))
            ax1.text(col-0.5, fila-0.5, f'#{i+1}', 
                    ha='center', va='center', color='blue', fontweight='bold', fontsize=12)
        
        # Gráfico de barras de seguridad
        rows_risk = prediction_result["row_risk_scores"]
        cols_risk = prediction_result["col_risk_scores"]
        
        x_pos = np.arange(5)
        width = 0.35
        
        ax2.bar(x_pos - width/2, rows_risk, width, label='Riesgo por Fila', alpha=0.7)
        ax2.bar(x_pos + width/2, cols_risk, width, label='Riesgo por Columna', alpha=0.7)
        
        ax2.set_xlabel('Posición')
        ax2.set_ylabel('Score de Riesgo')
        ax2.set_title('Análisis de Seguridad por Fila/Columna')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{i+1}' for i in range(5)])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Información adicional
        confidence = prediction_result["prediction_confidence"]
        total_prob = prediction_result["total_probability_sum"]
        
        fig.suptitle(f'Predicción de Minas | Confianza: {confidence:.3f} | Suma Total: {total_prob:.3f}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Visualización guardada en: {save_path}")
        
        plt.show()
        
        # Imprimir resumen
        print(f"\n Predicciones Top-{len(top_coords)}:")
        for i, (fila, col) in enumerate(top_coords):
            prob = prediction_result["top_m_probabilities"][i]
            print(f"   #{i+1}: Fila {fila}, Columna {col} (p={prob:.3f})")
        
        print(f"\n Zonas Seguras:")
        print(f"   Filas: {prediction_result['safe_rows']}")
        print(f"   Columnas: {prediction_result['safe_cols']}")

def main():
    """Función principal de inferencia"""
    print(" Iniciando predicción de minas")
    print("=" * 50)
    
    # Crear predictor
    predictor = MinesPredictor("model_mines.keras")
    
    try:
        # Realizar predicción
        result = predictor.predict_next_game(M=3)  # Ajustado para 3 minas
        
        # Mostrar resultado en formato JSON
        print("\n Resultado de predicción:")
        result_json = {
            "top_m_cells": result["top_m_cells"],
            "prob_matrix": result["prob_matrix"],
            "safe_rows": result["safe_rows"],
            "safe_cols": result["safe_cols"]
        }
        print(json.dumps(result_json, indent=2))
        
        # Visualizar
        predictor.visualize_prediction(result, save_path="prediction_visualization.png")
        
        print("\n Predicción completada exitosamente")
        
    except Exception as e:
        print(f" Error durante la predicción: {str(e)}")
        raise

if __name__ == "__main__":
    main()
