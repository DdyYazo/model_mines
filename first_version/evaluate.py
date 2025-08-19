# -*- coding: utf-8 -*-
"""
Script de Evaluación para Modelo Transformer de Minas
Cálculo de métricas, comparación con baseline y visualizaciones
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import MinesDataPipeline
from model_mines_transformer import MinesTransformer
from infer import MinesPredictor

class MinesEvaluator:
    """Evaluador completo para el modelo de predicción de minas"""
    
    def __init__(self, model_path: str = "model_mines.h5", data_dir: str = "games"):
        self.model_path = model_path
        self.data_dir = data_dir
        self.model = None
        self.pipeline = None
        self.data = None
        self.predictions = None
        self.baseline_predictions = None
        
    def load_model_and_data(self):
        """Carga modelo y datos de evaluación"""
        print(" Cargando modelo y datos...")
        
        # Cargar modelo
        if os.path.exists(self.model_path):
            self.model = MinesTransformer.load_model(self.model_path)
        else:
            print(f" Modelo no encontrado en {self.model_path}")
            return False
        
        # Cargar datos
        self.pipeline = MinesDataPipeline(data_dir=self.data_dir, sequence_length=3, random_seed=42)
        self.data = self.pipeline.process_full_pipeline()
        
        print(" Modelo y datos cargados correctamente")
        return True
    
    def generate_predictions(self):
        """Genera predicciones del modelo y baseline"""
        print(" Generando predicciones...")
        
        if self.model is None or self.data is None:
            raise ValueError("Modelo o datos no cargados")
        
        # Predicciones del modelo en conjunto de test
        self.predictions = self.model.model.predict(self.data['X_test'], verbose=0)
        
        # Baseline: usar frecuencias históricas
        baseline_freq = self.data['baseline_frequencies']
        self.baseline_predictions = np.tile(baseline_freq, (len(self.data['X_test']), 1))
        
        print(f" Predicciones generadas para {len(self.predictions)} muestras")
    
    def calculate_top_k_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> Dict[str, float]:
        """Calcula métricas Top-K específicas para minas"""
        
        hit_rates = []
        precisions = []
        recalls = []
        exact_matches = []
        
        for i in range(len(y_true)):
            # Posiciones reales de minas
            true_mines = set(np.where(y_true[i] == 1)[0])
            
            # Top-k predicciones
            top_k_pred = set(np.argsort(y_pred[i])[-k:])
            
            # Métricas
            intersection = true_mines & top_k_pred
            hits = len(intersection)
            
            hit_rates.append(hits / k)  # Proporción de aciertos en top-k
            precisions.append(hits / k)  # Precisión = hits / k predicciones
            recalls.append(hits / len(true_mines))  # Recall = hits / minas reales
            exact_matches.append(1.0 if hits == len(true_mines) else 0.0)
        
        f1_scores = []
        for p, r in zip(precisions, recalls):
            if p + r > 0:
                f1_scores.append(2 * p * r / (p + r))
            else:
                f1_scores.append(0.0)
        
        return {
            f'top_{k}_hit_rate': np.mean(hit_rates),
            f'precision_at_{k}': np.mean(precisions),
            f'recall_at_{k}': np.mean(recalls),
            f'f1_at_{k}': np.mean(f1_scores),
            f'exact_match_rate': np.mean(exact_matches)
        }
    
    def calculate_calibration_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calcula métricas de calibración de probabilidades"""
        
        # Brier Score
        brier_score = np.mean((y_pred - y_true) ** 2)
        
        # Curva de calibración (solo para visualización)
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calcular curva de calibración
        prob_true, prob_pred = calibration_curve(y_true_flat, y_pred_flat, n_bins=10)
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_flat > bin_lower) & (y_pred_flat <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true_flat[in_bin].mean()
                avg_confidence_in_bin = y_pred_flat[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'brier_score': brier_score,
            'expected_calibration_error': ece,
            'calibration_curve': (prob_true.tolist(), prob_pred.tolist())
        }
    
    def evaluate_comprehensive(self) -> Dict[str, Any]:
        """Evaluación completa del modelo"""
        print(" Realizando evaluación completa...")
        
        if self.predictions is None:
            self.generate_predictions()
        
        y_true = self.data['y_test']
        y_pred_model = self.predictions
        y_pred_baseline = self.baseline_predictions
        
        # Métricas Top-K para diferentes valores de K
        results = {}
        
        for k in [1, 2, 3, 4, 5]:
            model_metrics = self.calculate_top_k_metrics(y_true, y_pred_model, k)
            baseline_metrics = self.calculate_top_k_metrics(y_true, y_pred_baseline, k)
            
            # Agregar prefijos para distinguir
            for metric, value in model_metrics.items():
                results[f'model_{metric}'] = value
            for metric, value in baseline_metrics.items():
                results[f'baseline_{metric}'] = value
            
            # Calcular mejoras
            results[f'improvement_top_{k}_hit_rate'] = model_metrics[f'top_{k}_hit_rate'] - baseline_metrics[f'top_{k}_hit_rate']
        
        # Métricas de calibración
        calibration_metrics = self.calculate_calibration_metrics(y_true, y_pred_model)
        results.update(calibration_metrics)
        
        # Métricas adicionales
        results.update({
            'mean_prediction_sum': np.mean(np.sum(y_pred_model, axis=1)),  # Debe estar cerca de 3
            'std_prediction_sum': np.std(np.sum(y_pred_model, axis=1)),
            'baseline_mean_sum': np.mean(np.sum(y_pred_baseline, axis=1)),
        })
        
        print(" Evaluación completa terminada")
        return results
    
    def create_confusion_matrices(self) -> Dict[str, np.ndarray]:
        """Crea matrices de confusión para cada posición del tablero"""
        
        if self.predictions is None:
            self.generate_predictions()
        
        y_true = self.data['y_test']
        y_pred_model = (self.predictions > 0.5).astype(int)  # Threshold binario
        
        # Matriz de confusión global
        global_cm = confusion_matrix(y_true.flatten(), y_pred_model.flatten())
        
        # Matrices por posición del tablero
        position_cms = {}
        for pos in range(25):
            pos_cm = confusion_matrix(y_true[:, pos], y_pred_model[:, pos])
            position_cms[f'position_{pos}'] = pos_cm
        
        return {
            'global': global_cm,
            'by_position': position_cms
        }
    
    def plot_evaluation_results(self, results: Dict[str, Any], save_dir: str = "evaluation_plots"):
        """Genera visualizaciones de los resultados de evaluación"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Comparación Top-K Hit Rates
        self._plot_topk_comparison(results, save_dir)
        
        # 2. Curva de calibración
        self._plot_calibration_curve(results, save_dir)
        
        # 3. Distribución de probabilidades
        self._plot_probability_distributions(save_dir)
        
        # 4. Heatmap de errores por posición
        self._plot_position_error_heatmap(save_dir)
        
        print(f" Gráficos guardados en directorio: {save_dir}")
    
    def _plot_topk_comparison(self, results: Dict[str, Any], save_dir: str):
        """Gráfico de comparación Top-K entre modelo y baseline"""
        
        k_values = [1, 2, 3, 4, 5]
        model_hit_rates = [results[f'model_top_{k}_hit_rate'] for k in k_values]
        baseline_hit_rates = [results[f'baseline_top_{k}_hit_rate'] for k in k_values]
        
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(k_values))
        width = 0.35
        
        plt.bar(x - width/2, model_hit_rates, width, label='Modelo Transformer', alpha=0.8)
        plt.bar(x + width/2, baseline_hit_rates, width, label='Baseline (Frecuencias)', alpha=0.8)
        
        plt.xlabel('K (Top-K)')
        plt.ylabel('Hit Rate')
        plt.title('Comparación Top-K Hit Rate: Modelo vs Baseline')
        plt.xticks(x, [f'Top-{k}' for k in k_values])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for i, (model_val, baseline_val) in enumerate(zip(model_hit_rates, baseline_hit_rates)):
            plt.text(i - width/2, model_val + 0.01, f'{model_val:.3f}', ha='center', fontsize=9)
            plt.text(i + width/2, baseline_val + 0.01, f'{baseline_val:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'topk_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self, results: Dict[str, Any], save_dir: str):
        """Gráfico de curva de calibración"""
        
        prob_true, prob_pred = results['calibration_curve']
        
        plt.figure(figsize=(8, 8))
        
        plt.plot([0, 1], [0, 1], 'k:', label='Calibración perfecta')
        plt.plot(prob_pred, prob_true, 'o-', label=f'Modelo (ECE={results["expected_calibration_error"]:.3f})')
        
        plt.xlabel('Probabilidad predicha media')
        plt.ylabel('Fracción de positivos')
        plt.title('Curva de Calibración de Probabilidades')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'calibration_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_probability_distributions(self, save_dir: str):
        """Gráfico de distribuciones de probabilidades"""
        
        if self.predictions is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribución de probabilidades para celdas con minas
        y_true = self.data['y_test']
        mine_probs = self.predictions[y_true == 1]
        no_mine_probs = self.predictions[y_true == 0]
        
        # Histograma de probabilidades
        axes[0, 0].hist(mine_probs, bins=50, alpha=0.7, label='Celdas con minas', density=True)
        axes[0, 0].hist(no_mine_probs, bins=50, alpha=0.7, label='Celdas sin minas', density=True)
        axes[0, 0].set_xlabel('Probabilidad predicha')
        axes[0, 0].set_ylabel('Densidad')
        axes[0, 0].set_title('Distribución de Probabilidades por Tipo de Celda')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Suma de probabilidades por juego
        prob_sums = np.sum(self.predictions, axis=1)
        axes[0, 1].hist(prob_sums, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=3, color='red', linestyle='--', label='Target (3 minas)')
        axes[0, 1].set_xlabel('Suma de probabilidades por juego')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].set_title('Distribución de Suma de Probabilidades')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot por posición
        position_probs = [self.predictions[:, i] for i in range(25)]
        axes[1, 0].boxplot(position_probs, labels=[f'{i+1}' for i in range(25)])
        axes[1, 0].set_xlabel('Posición del tablero (1-25)')
        axes[1, 0].set_ylabel('Probabilidad predicha')
        axes[1, 0].set_title('Distribución de Probabilidades por Posición')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlación entre predicciones y verdad
        correlation_matrix = np.corrcoef(self.predictions.T, y_true.T)[:25, 25:]
        im = axes[1, 1].imshow(correlation_matrix.reshape(5, 5), cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 1].set_title('Correlación Predicciones vs Realidad por Posición')
        axes[1, 1].set_xlabel('Columna')
        axes[1, 1].set_ylabel('Fila')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'probability_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_position_error_heatmap(self, save_dir: str):
        """Heatmap de errores por posición del tablero"""
        
        if self.predictions is None:
            return
        
        y_true = self.data['y_test']
        
        # Calcular error cuadrático medio por posición
        mse_by_position = np.mean((self.predictions - y_true) ** 2, axis=0)
        mse_matrix = mse_by_position.reshape(5, 5)
        
        # Calcular bias por posición
        bias_by_position = np.mean(self.predictions - y_true, axis=0)
        bias_matrix = bias_by_position.reshape(5, 5)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MSE Heatmap
        sns.heatmap(mse_matrix, annot=True, fmt='.3f', cmap='Reds', 
                   square=True, ax=ax1, cbar_kws={'label': 'MSE'})
        ax1.set_title('Error Cuadrático Medio por Posición')
        ax1.set_xlabel('Columna')
        ax1.set_ylabel('Fila')
        
        # Bias Heatmap
        sns.heatmap(bias_matrix, annot=True, fmt='.3f', cmap='RdBu_r', 
                   square=True, ax=ax2, center=0, cbar_kws={'label': 'Bias'})
        ax2.set_title('Sesgo por Posición')
        ax2.set_xlabel('Columna')
        ax2.set_ylabel('Fila')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'position_error_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_evaluation_report(self, results: Dict[str, Any], save_path: str = "evaluation_report.json"):
        """Guarda reporte completo de evaluación"""
        
        # Agregar información adicional
        report = {
            'model_path': self.model_path,
            'data_info': self.data['pipeline_info'] if self.data else {},
            'evaluation_results': results,
            'summary': {
                'main_metric_top3_hit_rate': results.get('model_top_3_hit_rate', 0),
                'improvement_vs_baseline': results.get('improvement_top_3_hit_rate', 0),
                'exact_match_rate': results.get('model_exact_match_rate', 0),
                'calibration_quality': 'Good' if results.get('expected_calibration_error', 1) < 0.1 else 'Poor'
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f" Reporte de evaluación guardado en: {save_path}")
        return save_path

def main():
    """Función principal de evaluación"""
    print(" Iniciando evaluación completa del modelo")
    print("=" * 50)
    
    evaluator = MinesEvaluator("model_mines.h5", "games")
    
    try:
        # Cargar modelo y datos
        if not evaluator.load_model_and_data():
            print(" No se pudo cargar el modelo o los datos")
            return
        
        # Realizar evaluación
        results = evaluator.evaluate_comprehensive()
        
        # Generar visualizaciones
        evaluator.plot_evaluation_results(results)
        
        # Guardar reporte
        evaluator.save_evaluation_report(results)
        
        print("\n" + "=" * 50)
        print(" EVALUACIÓN COMPLETADA")
        print("=" * 50)
        print(f" Top-3 Hit Rate: {results['model_top_3_hit_rate']:.4f}")
        print(f" Mejora vs Baseline: {results['improvement_top_3_hit_rate']:.4f}")
        print(f" Exact Match Rate: {results['model_exact_match_rate']:.4f}")
        print(f" Brier Score: {results['brier_score']:.4f}")
        print(f" Calibration Error: {results['expected_calibration_error']:.4f}")
        
    except Exception as e:
        print(f" Error durante la evaluación: {str(e)}")
        raise

if __name__ == "__main__":
    main()
