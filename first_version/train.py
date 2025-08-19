# -*- coding: utf-8 -*-
"""
Script de Entrenamiento para Modelo Transformer de Minas
Incluye entrenamiento, validación, early stopping y guardado de modelo
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import MinesDataPipeline
from model_mines_transformer import create_default_model, MinesTransformer

# Configurar reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

class MinesTrainer:
    """Entrenador para el modelo Transformer de minas"""
    
    def __init__(self, 
                 sequence_length: int = 3,
                 data_dir: str = "games",
                 model_save_path: str = "model_mines.keras",
                 logs_dir: str = "training_logs"):
        
        self.sequence_length = sequence_length
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.logs_dir = logs_dir
        
        # Crear directorio de logs
        os.makedirs(logs_dir, exist_ok=True)
        
        self.pipeline = None
        self.model = None
        self.data = None
        self.history = None
        
    def prepare_data(self):
        """Prepara los datos usando el pipeline"""
        print(" Preparando datos...")
        
        self.pipeline = MinesDataPipeline(
            data_dir=self.data_dir,
            sequence_length=self.sequence_length,
            random_seed=42
        )
        
        self.data = self.pipeline.process_full_pipeline()
        
        # Calcular dimensiones para el modelo
        self.sequence_model_length = self.data['X_train'].shape[1]  # seq_len * 25
        self.feature_dim = self.data['X_train'].shape[2]
        
        print(f" Datos preparados:")
        print(f"    Secuencia modelo: {self.sequence_model_length}")
        print(f"    Feature dim: {self.feature_dim}")
        
    def create_model(self, **model_kwargs):
        """Crea el modelo Transformer"""
        print(" Creando modelo...")
        
        # Configuración optimizada para dataset pequeño
        model_config = {
            'sequence_length': self.sequence_model_length,
            'feature_dim': self.feature_dim,
            'embed_dim': 24,  # Reducido aún más
            'num_heads': 3,
            'ff_dim': 48,
            'num_transformer_blocks': 2,  # Muy reducido
            'dropout_rate': 0.3,  # Alto dropout
            'output_dim': 25
        }
        
        # Actualizar con parámetros personalizados
        model_config.update(model_kwargs)
        
        self.model = MinesTransformer(**model_config)
        self.model.build_model()
        self.model.compile_model(learning_rate=0.002)  # LR ligeramente mayor
        
        print(" Modelo creado")
        print(f"    Parámetros: {self.model.model.count_params():,}")
        
    def create_callbacks(self, patience: int = 15, save_best_only: bool = True):
        """Crea callbacks para el entrenamiento"""
        
        callbacks = []
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        callbacks.append(early_stopping)
        
        # Reduce Learning Rate on Plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        )
        callbacks.append(reduce_lr)
        
        # Model Checkpoint
        if save_best_only:
            checkpoint = ModelCheckpoint(
                self.model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                mode='min'
            )
            callbacks.append(checkpoint)
        
        return callbacks
    
    def train_model(self, 
                   epochs: int = 100,
                   batch_size: int = 16,  # Batch pequeño para dataset pequeño
                   patience: int = 20,
                   verbose: int = 1):
        """Entrena el modelo"""
        
        if self.model is None:
            raise ValueError("Modelo no creado. Llama create_model() primero.")
        
        if self.data is None:
            raise ValueError("Datos no preparados. Llama prepare_data() primero.")
        
        print(f" Iniciando entrenamiento...")
        print(f"    Epochs: {epochs}")
        print(f"    Batch size: {batch_size}")
        print(f"    Patience: {patience}")
        
        # Crear callbacks
        callbacks = self.create_callbacks(patience=patience)
        
        # Entrenar
        start_time = datetime.now()
        
        self.history = self.model.model.fit(
            self.data['X_train'], self.data['y_train'],
            validation_data=(self.data['X_val'], self.data['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds() / 60.0
        
        print(f" Entrenamiento completado en {training_time:.1f} minutos")
        
        # Guardar modelo en formato nativo Keras
        keras_path = self.model_save_path.replace('.h5', '.keras')
        self.model.model.save(keras_path)
        print(f" Modelo guardado en formato nativo: {keras_path}")
        
        # Guardar modelo final si no se guardó con checkpoint
        if not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
            self.model.save_model(self.model_save_path)
        
        return self.history
    
    def evaluate_model(self):
        """Evalúa el modelo en conjunto de test"""
        if self.model is None or self.data is None:
            raise ValueError("Modelo o datos no disponibles")
        
        print(" Evaluando modelo en conjunto de test...")
        
        test_results = self.model.model.evaluate(
            self.data['X_test'], self.data['y_test'],
            verbose=1
        )
        
        # Obtener predicciones
        predictions = self.model.model.predict(self.data['X_test'], verbose=0)
        
        # Calcular métricas adicionales
        metrics = self._calculate_additional_metrics(
            self.data['y_test'], 
            predictions,
            self.data['baseline_frequencies']
        )
        
        # Combinar resultados
        metric_names = self.model.model.metrics_names
        evaluation_results = dict(zip(metric_names, test_results))
        evaluation_results.update(metrics)
        
        print(" Resultados de evaluación:")
        for metric, value in evaluation_results.items():
            print(f"   {metric}: {value:.4f}")
        
        return evaluation_results, predictions
    
    def _calculate_additional_metrics(self, y_true, y_pred, baseline_freq):
        """Calcula métricas adicionales específicas para minas"""
        
        # Top-3 Hit Rate (métrica principal)
        hit_rates = []
        exact_matches = []
        
        for i in range(len(y_true)):
            # Posiciones reales de minas
            true_mines = np.where(y_true[i] == 1)[0]
            
            # Top-3 predicciones
            top_3_pred = np.argsort(y_pred[i])[-3:]
            
            # Hit rate: cuántas minas reales están en top-3
            hits = len(set(true_mines) & set(top_3_pred))
            hit_rates.append(hits / 3.0)
            
            # Match exacto
            exact_matches.append(1.0 if hits == 3 else 0.0)
        
        # Comparación con baseline (frecuencias históricas)
        baseline_top3 = np.argsort(baseline_freq)[-3:]
        baseline_hits = []
        
        for i in range(len(y_true)):
            true_mines = np.where(y_true[i] == 1)[0]
            hits = len(set(true_mines) & set(baseline_top3))
            baseline_hits.append(hits / 3.0)
        
        # Brier Score (calibración)
        brier_score = np.mean((y_pred - y_true) ** 2)
        
        return {
            'top3_hit_rate': np.mean(hit_rates),
            'exact_match_rate': np.mean(exact_matches),
            'baseline_hit_rate': np.mean(baseline_hits),
            'hit_rate_improvement': np.mean(hit_rates) - np.mean(baseline_hits),
            'brier_score': brier_score
        }
    
    def plot_training_curves(self, save_path: str = None):
        """Grafica curvas de entrenamiento"""
        if self.history is None:
            raise ValueError("No hay historial de entrenamiento disponible")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Binary Accuracy
        if 'binary_accuracy' in self.history.history:
            axes[0, 1].plot(self.history.history['binary_accuracy'], label='Train Acc')
            axes[0, 1].plot(self.history.history['val_binary_accuracy'], label='Val Acc')
            axes[0, 1].set_title('Binary Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Precision@3
        if 'precision_at_3' in self.history.history:
            axes[1, 0].plot(self.history.history['precision_at_3'], label='Train P@3')
            axes[1, 0].plot(self.history.history['val_precision_at_3'], label='Val P@3')
            axes[1, 0].set_title('Precision@3')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision@3')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # AUC
        if 'auc' in self.history.history:
            axes[1, 1].plot(self.history.history['auc'], label='Train AUC')
            axes[1, 1].plot(self.history.history['val_auc'], label='Val AUC')
            axes[1, 1].set_title('AUC')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Gráficos guardados en: {save_path}")
        
        plt.show()
        
    def save_training_report(self, evaluation_results: dict, save_path: str = None):
        """Guarda reporte de entrenamiento"""
        if save_path is None:
            save_path = os.path.join(self.logs_dir, f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report = {
            'model_config': {
                'sequence_length': self.sequence_length,
                'sequence_model_length': self.sequence_model_length,
                'feature_dim': self.feature_dim,
                'model_parameters': self.model.model.count_params() if self.model else None
            },
            'data_info': self.data['pipeline_info'] if self.data else {},
            'training_config': {
                'epochs_trained': len(self.history.history['loss']) if self.history else 0,
                'final_train_loss': float(self.history.history['loss'][-1]) if self.history else None,
                'final_val_loss': float(self.history.history['val_loss'][-1]) if self.history else None,
            },
            'evaluation_results': {k: float(v) for k, v in evaluation_results.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f" Reporte guardado en: {save_path}")
        return save_path

def main():
    """Función principal de entrenamiento"""
    print(" Iniciando entrenamiento del Modelo Transformer de Minas")
    print("=" * 60)
    
    # Configuración
    trainer = MinesTrainer(
        sequence_length=3,  # Usar 3 juegos como historia
        data_dir="games",
        model_save_path="model_mines.h5",
        logs_dir="training_logs"
    )
    
    try:
        # 1. Preparar datos
        trainer.prepare_data()
        
        # 2. Crear modelo
        trainer.create_model()
        
        # 3. Entrenar
        history = trainer.train_model(
            epochs=150,
            batch_size=8,  # Muy pequeño para dataset limitado
            patience=25,
            verbose=1
        )
        
        # 4. Evaluar
        evaluation_results, predictions = trainer.evaluate_model()
        
        # 5. Visualizar
        trainer.plot_training_curves(save_path="training_curves.png")
        
        # 6. Guardar reporte
        trainer.save_training_report(evaluation_results)
        
        print("\n" + "=" * 60)
        print(" ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f" Métrica principal (Top-3 Hit Rate): {evaluation_results['top3_hit_rate']:.4f}")
        print(f" Mejora vs Baseline: {evaluation_results['hit_rate_improvement']:.4f}")
        print(f" Matches exactos: {evaluation_results['exact_match_rate']:.4f}")
        print(f" Modelo guardado en: {trainer.model_save_path}")
        
    except Exception as e:
        print(f" Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main()
