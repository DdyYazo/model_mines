# Archivo: predict_game4.py
import os
from infer import MinesPredictor

def main():
    # 1. Crear archivo CSV con los datos históricos
    csv_content = """partida,columna,fila,mina
1,1,1,0
1,1,2,0
1,1,3,0
1,1,4,1
1,1,5,0
1,2,1,0
1,2,2,0
1,2,3,0
1,2,4,0
1,2,5,0
1,3,1,0
1,3,2,0
1,3,3,0
1,3,4,0
1,3,5,0
1,4,1,0
1,4,2,0
1,4,3,0
1,4,4,0
1,4,5,0
1,5,1,1
1,5,2,0
1,5,3,1
1,5,4,0
1,5,5,0
2,1,1,1
2,1,2,0
2,1,3,0
2,1,4,0
2,1,5,0
2,2,1,0
2,2,2,0
2,2,3,0
2,2,4,0
2,2,5,0
2,3,1,0
2,3,2,0
2,3,3,0
2,3,4,0
2,3,5,1
2,4,1,0
2,4,2,0
2,4,3,0
2,4,4,0
2,4,5,0
2,5,1,0
2,5,2,0
2,5,3,0
2,5,4,1
2,5,5,0
3,1,1,0
3,1,2,0
3,1,3,1
3,1,4,0
3,1,5,0
3,2,1,0
3,2,2,0
3,2,3,0
3,2,4,0
3,2,5,1
3,3,1,0
3,3,2,0
3,3,3,0
3,3,4,0
3,3,5,0
3,4,1,0
3,4,2,1
3,4,3,0
3,4,4,0
3,4,5,0
3,5,1,0
3,5,2,0
3,5,3,0
3,5,4,0
3,5,5,0"""
    
    # Guardar archivo temporal con formato de fecha válido
    temp_file = "minas-2025-08-15.csv"
    with open(temp_file, "w") as f:
        f.write(csv_content)
    
    try:
        # 2. Realizar predicción
        print("Cargando modelo y realizando prediccion...")
        predictor = MinesPredictor("model_mines_test.keras")
        result = predictor.predict_next_game(csv_path=temp_file, M=3)
        
        # 3. Mostrar resultados detallados
        print("\n" + "="*60)
        print("PREDICCION PARA LA PARTIDA 4 DE TU SESION ACTUAL")
        print("="*60)
        
        print("\nHistorial analizado:")
        print("  Partida 1: Minas en (1,4), (5,1), (5,3)")
        print("  Partida 2: Minas en (1,1), (3,5), (5,4)")
        print("  Partida 3: Minas en (1,3), (2,5), (4,2)")
        
        print(f"\nPREDICCION PARA PARTIDA 4:")
        print("Top-3 posiciones recomendadas para buscar minas:")
        
        for i, (col, fila) in enumerate(result["top_m_cells"]):
            prob = result["top_m_probabilities"][i]
            print(f"  #{i+1}: Columna {col}, Fila {fila}")
            print(f"       Probabilidad: {prob:.1%}")
        
        print(f"\nConfianza general: {result['prediction_confidence']:.1%}")
        
        print("\nEstrategia recomendada:")
        print("  Zonas PELIGROSAS (evitar):")
        danger_cells = result["top_m_cells"][:3]
        for col, fila in danger_cells:
            print(f"    - Columna {col}, Fila {fila}")
        
        print("  Zonas SEGURAS (preferir):")
        safe_rows = result["safe_rows"]
        safe_cols = result["safe_cols"]
        print(f"    - Filas mas seguras: {safe_rows}")
        print(f"    - Columnas mas seguras: {safe_cols}")
        
        # Matriz de probabilidades
        print("\nMatriz completa de probabilidades:")
        print("     Col1   Col2   Col3   Col4   Col5")
        prob_matrix = result["prob_matrix"]
        for i in range(5):
            row_str = f"Fila{i+1}"
            for j in range(5):
                row_str += f" {prob_matrix[i][j]:6.3f}"
            print(row_str)
        
        return result
        
    except Exception as e:
        print(f"Error durante la prediccion: {e}")
        return None
    
    finally:
        # Limpiar archivo temporal
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    main()