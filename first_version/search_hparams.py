# -*- coding: utf-8 -*-
"""
Búsqueda de Hiperparámetros con Optuna para Modelo Transformer de Minas
Optimización automática de configuración del modelo
"""

import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import MinesDataPipeline
from model_mines_transformer import MinesTransformer

# Configurar reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

class HyperparameterOptimizer:
    """Optimizador de hiperparámetros usando Optuna"""
    
    def __init__(self, data_dir: str = "games", n_trials: int = 50, timeout: int = 3600):
        self.data_dir = data_dir
        self.n_trials = n_trials
        self.timeout = timeout  # Tiempo límite en segundos
        
        # Cargar datos una sola vez
        self.pipeline = MinesDataPipeline(data_dir=data_dir, sequence_length=3, random_seed=42)
        self.data = self.pipeline.process_full_pipeline()
        
        print(f" Datos cargados: {len(self.data['X_train'])} muestras de entrenamiento")
    
    def objective(self, trial):
        """Función objetivo para optimizar"""
        
        # Espacios de búsqueda
        hyperparams = {
            'embed_dim': trial.suggest_categorical('embed_dim', [16, 24, 32, 48]),
            'num_heads': trial.suggest_categorical('num_heads', [2, 3, 4]),
            'ff_dim': trial.suggest_categorical('ff_dim', [32, 48, 64, 96]),
            'num_transformer_blocks': trial.suggest_int('num_transformer_blocks', 1, 4),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16]),
        }
        
        try:
            # Crear modelo con hiperparámetros sugeridos
            model = MinesTransformer(
                sequence_length=self.data['X_train'].shape[1],
                feature_dim=self.data['X_train'].shape[2],
                embed_dim=hyperparams['embed_dim'],
                num_heads=hyperparams['num_heads'],
                ff_dim=hyperparams['ff_dim'],
                num_transformer_blocks=hyperparams['num_transformer_blocks'],
                dropout_rate=hyperparams['dropout_rate'],
                output_dim=25
            )
            
            model.build_model()
            model.compile_model(learning_rate=hyperparams['learning_rate'])
            
            # Entrenar con early stopping agresivo
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )
            
            history = model.model.fit(
                self.data['X_train'], self.data['y_train'],
                validation_data=(self.data['X_val'], self.data['y_val']),
                epochs=50,  # Reducido para búsqueda rápida
                batch_size=hyperparams['batch_size'],
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluar en conjunto de validación
            val_loss = min(history.history['val_loss'])
            
            # Calcular métrica personalizada (Top-4 Hit Rate)
            predictions = model.model.predict(self.data['X_val'], verbose=0)
            hit_rate = self._calculate_hit_rate(self.data['y_val'], predictions, k=4)
            
            # Combinar métricas: minimizar pérdida y maximizar hit rate
            # Usar pérdida como métrica principal pero bonificar por hit rate
            objective_value = val_loss - 0.5 * hit_rate
            
            # Registrar métricas intermedias
            trial.set_user_attr('val_loss', val_loss)
            trial.set_user_attr('hit_rate', hit_rate)
            trial.set_user_attr('n_params', model.model.count_params())
            
            return objective_value
            
        except Exception as e:
            print(f" Error en trial {trial.number}: {str(e)}")
            return float('inf')  # Penalizar trials fallidos
    
    def _calculate_hit_rate(self, y_true, y_pred, k=4):
        """Calcula Top-K Hit Rate"""
        hit_rates = []
        
        for i in range(len(y_true)):
            # Posiciones reales de minas
            true_mines = set(np.where(y_true[i] == 1)[0])
            
            # Top-k predicciones
            top_k_pred = set(np.argsort(y_pred[i])[-k:])
            
            # Hit rate para esta muestra
            hits = len(true_mines & top_k_pred)
            hit_rates.append(hits / k)
        
        return np.mean(hit_rates)
    
    def optimize(self, study_name: str = "mines_transformer_optimization"):
        """Ejecutar optimización"""
        
        print(f" Iniciando búsqueda de hiperparámetros...")
        print(f"    Trials: {self.n_trials}")
        print(f"   ⏱ Timeout: {self.timeout}s")
        
        # Crear estudio
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Ejecutar optimización
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        print(f" Optimización completada:")
        print(f"    Mejor valor objetivo: {study.best_value:.4f}")
        print(f"    Mejores hiperparámetros:")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")
        
        # Métricas del mejor trial
        best_trial = study.best_trial
        print(f"    Val Loss: {best_trial.user_attrs.get('val_loss', 'N/A'):.4f}")
        print(f"    Hit Rate: {best_trial.user_attrs.get('hit_rate', 'N/A'):.4f}")
        print(f"    Parámetros: {best_trial.user_attrs.get('n_params', 'N/A'):,}")
        
        return study
    
    def save_results(self, study, save_path: str = "hyperparameter_optimization_results.json"):
        """Guardar resultados de optimización"""
        
        results = {
            'study_name': study.study_name,
            'n_trials': len(study.trials),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial_attrs': study.best_trial.user_attrs,
            'optimization_history': [
                {
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'user_attrs': trial.user_attrs
                } for trial in study.trials if trial.value is not None
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f" Resultados guardados en: {save_path}")
        return save_path
    
    def train_best_model(self, best_params: dict, save_path: str = "model_mines_optimized.h5"):
        """Entrenar modelo final con mejores hiperparámetros"""
        
        print(" Entrenando modelo final con mejores hiperparámetros...")
        
        # Crear modelo optimizado
        model = MinesTransformer(
            sequence_length=self.data['X_train'].shape[1],
            feature_dim=self.data['X_train'].shape[2],
            embed_dim=best_params['embed_dim'],
            num_heads=best_params['num_heads'],
            ff_dim=best_params['ff_dim'],
            num_transformer_blocks=best_params['num_transformer_blocks'],
            dropout_rate=best_params['dropout_rate'],
            output_dim=25
        )
        
        model.build_model()
        model.compile_model(learning_rate=best_params['learning_rate'])
        
        # Entrenar por más épocas
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        history = model.model.fit(
            self.data['X_train'], self.data['y_train'],
            validation_data=(self.data['X_val'], self.data['y_val']),
            epochs=100,
            batch_size=best_params['batch_size'],
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Guardar modelo optimizado
        model.save_model(save_path)
        
        # Evaluar en test
        test_results = model.model.evaluate(self.data['X_test'], self.data['y_test'], verbose=0)
        predictions = model.model.predict(self.data['X_test'], verbose=0)
        hit_rate = self._calculate_hit_rate(self.data['y_test'], predictions, k=4)
        
        print(f" Resultados finales del modelo optimizado:")
        print(f"    Test Loss: {test_results[0]:.4f}")
        print(f"    Test Hit Rate: {hit_rate:.4f}")
        
        return model, history

def main():
    """Función principal de optimización"""
    print(" Iniciando Optimización de Hiperparámetros")
    print("=" * 60)
    
    # Configuración de optimización
    optimizer = HyperparameterOptimizer(
        data_dir="games",
        n_trials=30,  # Reducido para prueba rápida
        timeout=1800  # 30 minutos
    )
    
    try:
        # Ejecutar optimización
        study = optimizer.optimize()
        
        # Guardar resultados
        results_path = optimizer.save_results(study)
        
        # Entrenar modelo final
        best_model, history = optimizer.train_best_model(
            study.best_params,
            save_path="model_mines_optimized.h5"
        )
        
        print("\n" + "=" * 60)
        print(" OPTIMIZACIÓN COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        print(f" Resultados: {results_path}")
        print(f" Modelo optimizado: model_mines_optimized.h5")
        
        # Mostrar top 3 configuraciones
        print(f"\n Top 3 configuraciones:")
        sorted_trials = sorted([t for t in study.trials if t.value is not None], 
                             key=lambda x: x.value)[:3]
        
        for i, trial in enumerate(sorted_trials, 1):
            print(f"\n#{i} - Valor objetivo: {trial.value:.4f}")
            print(f"    Hit Rate: {trial.user_attrs.get('hit_rate', 'N/A'):.4f}")
            print(f"    Parámetros: {trial.params}")
        
    except Exception as e:
        print(f" Error durante la optimización: {str(e)}")
        raise

if __name__ == "__main__":
    main()
