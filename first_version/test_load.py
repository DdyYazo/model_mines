#!/usr/bin/env python3
"""
Test script para verificar carga del modelo
"""

import sys
import os

print(" Testing model loading...")

try:
    # Import modules
    from model_mines_transformer import MinesTransformer
    print(" Model module imported successfully")
    
    # Load model
    model = MinesTransformer.load_model('model_mines.keras')
    print(" Model loaded successfully")
    
    # Check model properties
    print(f" Model input shape: {model.model.input_shape}")
    print(f" Model output shape: {model.model.output_shape}")
    
    print(" All tests passed!")
    
except Exception as e:
    print(f" Error: {str(e)}")
    import traceback
    traceback.print_exc()
