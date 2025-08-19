# -*- coding: utf-8 -*-
"""
Data Pipeline para Modelo Transformer de Predicción de Minas
Maneja carga, validación, feature engineering y splits temporales
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class MinesDataPipeline:
    def __init__(self, data_dir: str = "games", sequence_length: int = 5, random_seed: int = 42):
        """
        Inicializa el pipeline de datos para minas
        
        Args:
            data_dir: Directorio con archivos CSV
            sequence_length: Longitud de secuencia temporal para el modelo
            random_seed: Semilla para reproducibilidad
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        self.scaler = None
        self.feature_dim = None
        
        np.random.seed(random_seed)
    
    def _calculate_streak(self, values: np.ndarray, value: int) -> np.ndarray:
        """Calcula rachas consecutivas de un valor específico"""
        streaks = np.zeros(len(values))
        current_streak = 0
        
        for i, val in enumerate(values):
            if val == value:
                current_streak += 1
                streaks[i] = current_streak
            else:
                current_streak = 0
                streaks[i] = 0
        
        return streaks
        
    def load_and_validate_data(self) -> pd.DataFrame:
        """Carga y valida todos los archivos CSV"""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No se encontraron archivos CSV en {self.data_dir}")
        
        dfs = []
        total_games = 0
        
        print(f"Cargando {len(csv_files)} archivos CSV...")
        
        for file in sorted(csv_files):
            file_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(file_path)
            
            # Validación básica
            required_cols = {'partida', 'columna', 'fila', 'mina'}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Archivo {file} no tiene las columnas requeridas: {required_cols}")
            
            # Verificar rango de coordenadas
            if not (df['columna'].between(1, 5).all() and df['fila'].between(1, 5).all()):
                raise ValueError(f"Coordenadas fuera de rango 1-5 en {file}")
            
            # Verificar valores binarios de mina
            if not df['mina'].isin([0, 1]).all():
                raise ValueError(f"Valores de mina no binarios en {file}")
            
            # Extraer fecha del nombre del archivo para ordenamiento temporal
            date_str = file.replace('minas-', '').replace('.csv', '')
            # Manejar sufijos como _p2, _p3, etc.
            if '_p' in date_str:
                date_str = date_str.split('_p')[0]
            df['fecha'] = pd.to_datetime(date_str)
            df['archivo'] = file
            
            # Verificar distribución de minas por partida
            minas_por_partida = df.groupby('partida')['mina'].sum()
            distribucion = minas_por_partida.value_counts().sort_index().to_dict()
            print(f"[{file}] Distribución de minas por partida: {distribucion}")
            
            # Detectar M predominante
            minas_predominantes = minas_por_partida.mode()[0]
            print(f"    Minas predominantes: {minas_predominantes}")
            
            games_in_file = df['partida'].nunique()
            total_games += games_in_file
            print(f"   OK {file}: {games_in_file} partidas")
            
            dfs.append(df)
        
        # Consolidar datos
        df_consolidated = pd.concat(dfs, ignore_index=True)
        df_consolidated = df_consolidated.sort_values(['fecha', 'archivo', 'partida', 'columna', 'fila'])
        
        print(f"Total: {total_games} partidas cargadas")
        print(f"Fechas: {df_consolidated['fecha'].min()} -> {df_consolidated['fecha'].max()}")
        
        return df_consolidated
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ingeniería de características avanzada"""
        print("Generando features...")
        
        features_list = []
        
        # Procesar por partida individual
        for (fecha, archivo, partida), group in df.groupby(['fecha', 'archivo', 'partida']):
            
            # Crear matriz 5x5 del estado actual
            game_matrix = np.zeros((5, 5))
            for _, row in group.iterrows():
                game_matrix[row['fila']-1, row['columna']-1] = row['mina']
            
            # Features básicas por celda
            for fila in range(1, 6):
                for col in range(1, 6):
                    features = {
                        'fecha': fecha,
                        'archivo': archivo, 
                        'partida': partida,
                        'fila': fila,
                        'columna': col,
                        'target': game_matrix[fila-1, col-1],
                        
                        # Coordenadas normalizadas
                        'fila_norm': (fila - 1) / 4.0,
                        'col_norm': (col - 1) / 4.0,
                        
                        # Posición en el tablero (embeddings espaciales)
                        'pos_radial': np.sqrt((fila-3)**2 + (col-3)**2) / np.sqrt(8),  # Distancia al centro
                        'pos_diagonal': abs(fila - col) / 4.0,  # Posición diagonal
                        'pos_border': 1.0 if (fila in [1,5] or col in [1,5]) else 0.0,  # Borde del tablero
                        'pos_corner': 1.0 if (fila in [1,5] and col in [1,5]) else 0.0,  # Esquina
                        'pos_center': 1.0 if (fila == 3 and col == 3) else 0.0,  # Centro
                        
                        # One-hot encoding de posición
                        'pos_index': (fila-1)*5 + (col-1),  # Índice único 0-24
                    }
                    
                    # Embeddings posicionales sinusoidales
                    pos_idx = features['pos_index']
                    for i in range(8):  # 8 dimensiones de embedding
                        if i % 2 == 0:
                            features[f'pos_sin_{i}'] = np.sin(pos_idx / (10000 ** (i / 8)))
                        else:
                            features[f'pos_cos_{i}'] = np.cos(pos_idx / (10000 ** ((i-1) / 8)))
                    
                    features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        features_df = features_df.sort_values(['fecha', 'archivo', 'partida', 'fila', 'columna'])
        
        # Features históricos y temporales
        print("Calculando features historicos...")
        features_df = self._add_historical_features(features_df)
        
        print(f"Features generados: {len(features_df)} registros")
        return features_df
    
    def _add_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade características históricas y de momentum para capturar patrones RNG"""
        
        # Ordenar temporalmente
        df = df.sort_values(['fecha', 'archivo', 'partida', 'pos_index'])
        
        # Calcular estadísticas históricas por posición
        historical_stats = []
        
        for pos_idx in range(25):  # 25 posiciones en tablero 5x5
            pos_data = df[df['pos_index'] == pos_idx].copy()
            
            # Frecuencia histórica acumulativa
            pos_data['hist_freq'] = pos_data['target'].expanding().mean()
            
            # Suavizado exponencial (momentum corto y largo plazo)
            pos_data['hist_smooth_short'] = pos_data['target'].ewm(alpha=0.5).mean()  # Rápida adaptación
            pos_data['hist_smooth_long'] = pos_data['target'].ewm(alpha=0.1).mean()   # Tendencia estable
            
            # Contadores
            pos_data['hist_count_total'] = range(1, len(pos_data) + 1)
            pos_data['hist_count_mines'] = pos_data['target'].expanding().sum()
            
            # Tendencias recientes (ventanas múltiples)
            pos_data['hist_trend_3'] = pos_data['target'].rolling(window=3, min_periods=1).mean()
            pos_data['hist_trend_5'] = pos_data['target'].rolling(window=5, min_periods=1).mean()
            pos_data['hist_trend_10'] = pos_data['target'].rolling(window=10, min_periods=1).mean()
            
            # Patrones de racha (detectar secuencias de minas/no-minas)
            pos_data['streak_mines'] = self._calculate_streak(pos_data['target'].values, value=1)
            pos_data['streak_no_mines'] = self._calculate_streak(pos_data['target'].values, value=0)
            
            # Variabilidad histórica (detecta cambios en comportamiento RNG)
            pos_data['hist_variance'] = pos_data['target'].rolling(window=10, min_periods=3).var().fillna(0)
            pos_data['hist_volatility'] = pos_data['target'].rolling(window=10, min_periods=3).std().fillna(0)
            
            # Patrones cíclicos (detectar periodicidades en RNG)
            pos_data['cycle_mod_3'] = (pos_data.index % 3).astype(float) / 2.0
            pos_data['cycle_mod_5'] = (pos_data.index % 5).astype(float) / 4.0
            pos_data['cycle_mod_7'] = (pos_data.index % 7).astype(float) / 6.0
            
            # Correlación temporal (patron de auto-correlación)
            if len(pos_data) >= 5:
                lag_1 = pos_data['target'].shift(1).fillna(0)
                lag_2 = pos_data['target'].shift(2).fillna(0)
                pos_data['autocorr_lag1'] = pos_data['target'] * lag_1
                pos_data['autocorr_lag2'] = pos_data['target'] * lag_2
            else:
                pos_data['autocorr_lag1'] = 0.0
                pos_data['autocorr_lag2'] = 0.0
            
            historical_stats.append(pos_data)
        
        df_with_hist = pd.concat(historical_stats).sort_values(['fecha', 'archivo', 'partida', 'pos_index'])
        
        # Features de contexto del juego actual y patrones RNG avanzados
        game_features = []
        for (fecha, archivo, partida), game_group in df_with_hist.groupby(['fecha', 'archivo', 'partida']):
            game_data = game_group.copy()
            
            # Estadísticas del tablero actual
            total_mines = game_data['target'].sum()
            game_data['game_total_mines'] = total_mines
            game_data['game_density'] = total_mines / 25.0
            
            # Distribución por filas y columnas
            for fila in range(1, 6):
                mines_in_row = game_data[game_data['fila'] == fila]['target'].sum()
                game_data.loc[game_data['fila'] == fila, 'row_mines_count'] = mines_in_row
                game_data.loc[game_data['fila'] == fila, 'row_density'] = mines_in_row / 5.0
                
            for col in range(1, 6):
                mines_in_col = game_data[game_data['columna'] == col]['target'].sum()
                game_data.loc[game_data['columna'] == col, 'col_mines_count'] = mines_in_col
                game_data.loc[game_data['columna'] == col, 'col_density'] = mines_in_col / 5.0
            
            # Patrones espaciales del tablero actual
            game_matrix = game_data['target'].values.reshape(5, 5)
            
            # Añadir features de vecindad para cada celda
            for idx, row in game_data.iterrows():
                fila_idx, col_idx = int(row['fila']) - 1, int(row['columna']) - 1
                
                # Conteo de vecinos con minas (8-conectividad)
                neighbors_sum = 0
                neighbors_count = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = fila_idx + di, col_idx + dj
                        if 0 <= ni < 5 and 0 <= nj < 5:
                            neighbors_sum += game_matrix[ni, nj]
                            neighbors_count += 1
                
                game_data.loc[idx, 'neighbors_mines'] = neighbors_sum
                game_data.loc[idx, 'neighbors_density'] = neighbors_sum / neighbors_count if neighbors_count > 0 else 0
                
                # Distancia a la mina más cercana
                mine_positions = np.argwhere(game_matrix == 1)
                if len(mine_positions) > 0:
                    distances = [np.sqrt((fila_idx - mp[0])**2 + (col_idx - mp[1])**2) for mp in mine_positions]
                    game_data.loc[idx, 'dist_nearest_mine'] = min(distances) / np.sqrt(32)  # Normalizado
                else:
                    game_data.loc[idx, 'dist_nearest_mine'] = 1.0
            
            # Features de archivo/sesión (para detectar patrones por archivo)
            game_data['file_hash'] = hash(archivo) % 100 / 100.0  # Hash normalizado del archivo
            game_data['game_in_file'] = partida / 50.0  # Número de partida normalizado en el archivo
            
            game_features.append(game_data)
        
        final_df = pd.concat(game_features).sort_values(['fecha', 'archivo', 'partida', 'pos_index'])
        
        # Llenar valores NaN
        for col in final_df.select_dtypes(include=[np.number]).columns:
            final_df[col] = final_df[col].fillna(0.0)
            
        return final_df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List]:
        """Crea secuencias temporales para el modelo Transformer"""
        print(f"Creando secuencias de longitud {self.sequence_length}...")
        
        # Seleccionar features numéricas para el modelo (incluyendo nuevas features RNG)
        feature_cols = [col for col in df.columns if col.startswith(('pos_', 'hist_', 'fila_norm', 
                                                                    'col_norm', 'game_', 'row_', 'col_',
                                                                    'neighbors_', 'dist_', 'file_', 'streak_',
                                                                    'cycle_', 'autocorr_'))]
        
        # Asegurar que tenemos las features principales
        required_features = ['pos_index', 'fila_norm', 'col_norm'] + [f'pos_sin_{i}' for i in range(0, 8, 2)] + [f'pos_cos_{i}' for i in range(1, 8, 2)]
        for feat in required_features:
            if feat not in feature_cols:
                feature_cols.append(feat)
        
        self.feature_dim = len(feature_cols)
        print(f"Dimension de features: {self.feature_dim}")
        print(f"Features utilizadas: {feature_cols[:10]}... (+{len(feature_cols)-10})")
        
        # Preparar datos por juego completo (25 celdas)
        games_data = []
        games_targets = []
        games_metadata = []
        
        for (fecha, archivo, partida), game_group in df.groupby(['fecha', 'archivo', 'partida']):
            # Ordenar por posición para mantener orden del tablero
            game_group = game_group.sort_values('pos_index')
            
            if len(game_group) == 25:  # Juego completo
                # Features del juego (25 celdas x feature_dim)
                game_features = game_group[feature_cols].values  # Shape: (25, feature_dim)
                
                # Target del juego (25 celdas)
                game_target = game_group['target'].values  # Shape: (25,)
                
                games_data.append(game_features)
                games_targets.append(game_target)
                games_metadata.append({
                    'fecha': fecha,
                    'archivo': archivo,
                    'partida': partida
                })
        
        print(f"Total juegos validos: {len(games_data)}")
        
        if len(games_data) < self.sequence_length + 1:
            raise ValueError(f"Datos insuficientes: {len(games_data)} juegos < {self.sequence_length + 1} requeridos")
        
        # Crear secuencias temporales
        X_sequences = []
        y_sequences = []
        metadata_sequences = []
        
        for i in range(len(games_data) - self.sequence_length):
            # Secuencia de entrada: últimos N juegos
            sequence_features = np.array(games_data[i:i + self.sequence_length])  # Shape: (seq_len, 25, feature_dim)
            
            # Target: siguiente juego
            target = games_targets[i + self.sequence_length]  # Shape: (25,)
            
            X_sequences.append(sequence_features)
            y_sequences.append(target)
            metadata_sequences.append(games_metadata[i + self.sequence_length])
        
        X = np.array(X_sequences, dtype=np.float32)  # Shape: (n_samples, seq_len, 25, feature_dim)
        y = np.array(y_sequences, dtype=np.float32)  # Shape: (n_samples, 25)
        
        # Reshape X para el modelo: (n_samples, seq_len * 25, feature_dim)
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2], X.shape[3])
        
        print(f"Secuencias creadas:")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        
        return X, y, metadata_sequences
    
    def temporal_split(self, X: np.ndarray, y: np.ndarray, metadata: List, 
                      train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        """División temporal estricta para evitar data leakage"""
        print(" Realizando split temporal...")
        
        # Ordenar por fecha
        dates = [meta['fecha'] for meta in metadata]
        sorted_indices = np.argsort(dates)
        
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        metadata_sorted = [metadata[i] for i in sorted_indices]
        
        n_samples = len(X_sorted)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        splits = {
            'X_train': X_sorted[:train_end],
            'y_train': y_sorted[:train_end],
            'metadata_train': metadata_sorted[:train_end],
            
            'X_val': X_sorted[train_end:val_end],
            'y_val': y_sorted[train_end:val_end],
            'metadata_val': metadata_sorted[train_end:val_end],
            
            'X_test': X_sorted[val_end:],
            'y_test': y_sorted[val_end:],
            'metadata_test': metadata_sorted[val_end:]
        }
        
        print(f"Split temporal completado:")
        print(f"    Train: {len(splits['X_train'])} secuencias")
        print(f"    Val:   {len(splits['X_val'])} secuencias") 
        print(f"    Test:  {len(splits['X_test'])} secuencias")
        
        # Mostrar rango de fechas
        if splits['metadata_train']:
            train_dates = [m['fecha'] for m in splits['metadata_train']]
            print(f"    Train: {min(train_dates)} -> {max(train_dates)}")
        
        if splits['metadata_val']:
            val_dates = [m['fecha'] for m in splits['metadata_val']]
            print(f"    Val:   {min(val_dates)} -> {max(val_dates)}")
            
        if splits['metadata_test']:
            test_dates = [m['fecha'] for m in splits['metadata_test']]
            print(f"    Test:  {min(test_dates)} -> {max(test_dates)}")
        
        return splits
    
    def get_baseline_frequencies(self, y_train: np.ndarray) -> np.ndarray:
        """Calcula frecuencias históricas como baseline"""
        return np.mean(y_train, axis=0)
    
    def process_full_pipeline(self) -> Dict:
        """Ejecuta todo el pipeline de datos"""
        print(" Iniciando pipeline completo de datos...")
        
        # 1. Cargar y validar
        df_raw = self.load_and_validate_data()
        
        # 2. Feature engineering
        df_features = self.create_features(df_raw)
        
        # 3. Crear secuencias
        X, y, metadata = self.create_sequences(df_features)
        
        # 4. Split temporal
        splits = self.temporal_split(X, y, metadata)
        
        # 5. Baseline
        baseline_freq = self.get_baseline_frequencies(splits['y_train'])
        splits['baseline_frequencies'] = baseline_freq
        
        # 6. Guardar info del pipeline
        splits['pipeline_info'] = {
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'random_seed': self.random_seed,
            'total_samples': len(X),
            'total_games_loaded': len(df_raw) // 25
        }
        
        print("Pipeline de datos completado!")
        return splits

if __name__ == "__main__":
    # Prueba del pipeline
    pipeline = MinesDataPipeline(sequence_length=3)
    data = pipeline.process_full_pipeline()
    print("\nResumen final:")
    for key, value in data['pipeline_info'].items():
        print(f"   {key}: {value}")
