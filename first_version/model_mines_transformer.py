# -*- coding: utf-8 -*-
"""
Modelo Transformer para Predicción de Minas
Implementación optimizada en Keras/TensorFlow 2.x
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Optional, Dict, Any

# Configurar TensorFlow para reproducibilidad
tf.random.set_seed(42)

def custom_mines_loss(y_true, y_pred):
    """Función de pérdida personalizada: BCE + regularización para M=3 minas"""
    # Binary Cross Entropy para cada celda
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_loss = tf.reduce_mean(bce)
    
    # Regularización: penalizar si la suma de probabilidades no está cerca de 3
    prob_sum = tf.reduce_sum(y_pred, axis=1)  # Suma de probabilidades por muestra
    target_sum = 3.0  # Esperamos exactamente 3 minas
    sum_penalty = tf.reduce_mean(tf.square(prob_sum - target_sum))
    
    # Pérdida total con mayor peso en la regularización para mayor precisión
    total_loss = bce_loss + 0.15 * sum_penalty
    return total_loss

def precision_at_3(y_true, y_pred):
    """Precision@3: proporción de minas reales en el top-3 predicho"""
    # Obtener top-3 predicciones
    top_3_indices = tf.nn.top_k(y_pred, k=3).indices
    
    # Crear máscara binaria para top-3
    batch_size = tf.shape(y_pred)[0]
    indices = tf.stack([
        tf.repeat(tf.range(batch_size), 3),
        tf.reshape(top_3_indices, [-1])
    ], axis=1)
    
    top_3_binary = tf.scatter_nd(
        indices, 
        tf.ones(batch_size * 3), 
        tf.shape(y_pred)
    )
    
    # Calcular intersección
    intersection = tf.reduce_sum(y_true * top_3_binary, axis=1)
    precision = tf.reduce_mean(intersection / 3.0)
    return precision

def recall_at_3(y_true, y_pred):
    """Recall@3: proporción de minas reales encontradas en top-3"""
    # Obtener top-3 predicciones
    top_3_indices = tf.nn.top_k(y_pred, k=3).indices
    
    # Crear máscara binaria para top-3
    batch_size = tf.shape(y_pred)[0]
    indices = tf.stack([
        tf.repeat(tf.range(batch_size), 3),
        tf.reshape(top_3_indices, [-1])
    ], axis=1)
    
    top_3_binary = tf.scatter_nd(
        indices, 
        tf.ones(batch_size * 3), 
        tf.shape(y_pred)
    )
    
    # Calcular intersección
    intersection = tf.reduce_sum(y_true * top_3_binary, axis=1)
    total_mines = tf.reduce_sum(y_true, axis=1)
    recall = tf.reduce_mean(intersection / tf.maximum(total_mines, 1.0))
    return recall

class PositionalEncoding(layers.Layer):
    """Codificación posicional sinusoidal para Transformer"""
    
    def __init__(self, sequence_length: int, embed_dim: int, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        # Crear matriz de codificación posicional
        position = np.arange(self.sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        
        pos_encoding = np.zeros((self.sequence_length, self.embed_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.Variable(
            initial_value=pos_encoding.astype('float32'),
            trainable=False,
            name="positional_encoding"
        )
        super().build(input_shape)
        
    def call(self, x):
        """Añade codificación posicional a la entrada"""
        return x + self.pos_encoding
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim,
        })
        return config

class TransformerBlock(layers.Layer):
    """Bloque Transformer con Multi-Head Attention y Feed Forward"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Multi-Head Attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim,
            dropout=dropout_rate
        )
        
        # Feed Forward Network
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
        ])
        
        # Layer Normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def build(self, input_shape):
        """Build method for proper layer initialization"""
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass del bloque Transformer"""
        # Multi-Head Attention con conexión residual
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed Forward Network con conexión residual
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

class MinesTransformer:
    """Modelo Transformer para predicción de minas"""
    
    def __init__(self, 
                 sequence_length: int = 125,  # 5 juegos × 25 celdas
                 feature_dim: int = 20,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 ff_dim: int = 128,
                 num_transformer_blocks: int = 4,
                 dropout_rate: float = 0.15,
                 output_dim: int = 25):  # 25 celdas del tablero
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        
        self.model = None
        
    def build_model(self) -> Model:
        """Construye el modelo Transformer"""
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.feature_dim), name="sequence_input")
        
        # Proyección a dimensión de embedding
        x = layers.Dense(self.embed_dim, activation="relu", name="input_projection")(inputs)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Codificación posicional
        x = PositionalEncoding(self.sequence_length, self.embed_dim)(x)
        
        # Stack de bloques Transformer
        for i in range(self.num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                name=f"transformer_block_{i}"
            )(x, training=True)
        
        # Agregación: usar las últimas 25 posiciones (último juego) + promedio temporal
        # Opción 1: Solo último juego
        last_game_features = x[:, -25:, :]  # Shape: (batch, 25, embed_dim)
        
        # Opción 2: Pooling temporal de toda la secuencia  
        temporal_features = layers.GlobalAveragePooling1D(name="temporal_pooling")(x)  # Shape: (batch, embed_dim)
        
        # Reshape las features del último juego para combinar con pooling temporal
        last_game_flat = layers.Flatten(name="last_game_flatten")(last_game_features)  # Shape: (batch, 25*embed_dim)
        
        # Concatenar features temporales y espaciales
        combined_features = layers.Concatenate(name="combine_features")([last_game_flat, temporal_features])
        
        # Cabeza de clasificación mejorada con batch normalization
        x = layers.Dense(384, activation="relu", name="classification_head_1")(combined_features)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(192, activation="relu", name="classification_head_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.Dropout(self.dropout_rate * 0.8)(x)  # Dropout menor en capas profundas
        
        x = layers.Dense(96, activation="relu", name="classification_head_3")(x)
        x = layers.Dropout(self.dropout_rate * 0.6)(x)
        
        # Output layer: 25 probabilidades (una por celda)
        outputs = layers.Dense(self.output_dim, activation="sigmoid", name="mine_probabilities")(x)
        
        # Crear modelo
        model = Model(inputs=inputs, outputs=outputs, name="MinesTransformer")
        
        self.model = model
        return model
    
    def compile_model(self, 
                     learning_rate: float = 0.001,
                     loss_weights: Optional[Dict[str, float]] = None) -> None:
        """Compila el modelo con función de pérdida personalizada"""
        
        if self.model is None:
            raise ValueError("Modelo no construido. Llama build_model() primero.")
        
        # Compilar modelo
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=custom_mines_loss,
            metrics=[
                'binary_accuracy',
                precision_at_3,
                recall_at_3,
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        print(f"Modelo compilado con learning_rate={learning_rate}")
        
    def get_model_summary(self) -> str:
        """Retorna resumen del modelo"""
        if self.model is None:
            return "Modelo no construido"
        
        # Crear string buffer para capturar la salida
        import io
        buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        summary = buffer.getvalue()
        buffer.close()
        
        return summary
    
    def save_model(self, filepath: str) -> None:
        """Guarda el modelo entrenado"""
        if self.model is None:
            raise ValueError("Modelo no construido")
        
        self.model.save(filepath)
        print(f" Modelo guardado en: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MinesTransformer':
        """Carga un modelo guardado"""
        
        # Registrar clases personalizadas
        custom_objects = {
            'PositionalEncoding': PositionalEncoding,
            'TransformerBlock': TransformerBlock,
            'custom_mines_loss': custom_mines_loss,
            'precision_at_3': precision_at_3,
            'recall_at_3': recall_at_3
        }
        
        loaded_model = keras.models.load_model(filepath, custom_objects=custom_objects)
        
        # Crear instancia de la clase
        instance = cls()
        instance.model = loaded_model
        
        print(f" Modelo cargado desde: {filepath}")
        return instance

def create_default_model(sequence_length: int, feature_dim: int) -> MinesTransformer:
    """Crea modelo con configuración optimizada para dataset de 3 minas"""
    
    model = MinesTransformer(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        embed_dim=48,  # Aumentado para mejor representación de features RNG
        num_heads=6,   # Más heads para capturar patrones complejos
        ff_dim=96,     # Proporción 2:1 con embed_dim
        num_transformer_blocks=4,  # Aumentado para mayor capacidad
        dropout_rate=0.25,  # Regularización moderada
        output_dim=25
    )
    
    model.build_model()
    model.compile_model(learning_rate=0.0008)  # Learning rate ligeramente menor
    
    return model

if __name__ == "__main__":
    # Prueba del modelo
    print("Probando modelo Transformer...")
    
    # Crear modelo de prueba
    model = create_default_model(sequence_length=75, feature_dim=30)  # 3 juegos × 25 celdas, features expandidas
    
    print("\n Resumen del modelo:")
    print(model.get_model_summary())
    
    # Probar con datos sintéticos
    batch_size = 8
    X_test = np.random.random((batch_size, 75, 15)).astype('float32')
    y_test = np.random.randint(0, 2, (batch_size, 25)).astype('float32')
    
    print(f"\n Probando predicción con datos sintéticos:")
    print(f"   Input shape: {X_test.shape}")
    print(f"   Output shape: {y_test.shape}")
    
    predictions = model.model.predict(X_test, verbose=0)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Predictions sample: {predictions[0][:5]}...")
    
    print(" Modelo funcionando correctamente!")
