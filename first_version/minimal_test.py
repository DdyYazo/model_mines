#!/usr/bin/env python3
"""
Minimal model loading test
"""

print(" Starting minimal model test...")

try:
    print(" Importing model module...")
    from model_mines_transformer import MinesTransformer
    print(" Module imported")
    
    print(" Loading model...")
    model = MinesTransformer.load_model('model_mines.keras')
    print(" Model loaded")
    
    print(" Model info:")
    print(f"  - Input shape: {model.model.input_shape}")
    print(f"  - Output shape: {model.model.output_shape}")
    
    print(" Test completed successfully!")
    
except Exception as e:
    print(f" Error: {e}")
    import traceback
    traceback.print_exc()
