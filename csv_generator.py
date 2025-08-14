#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

def generar_mapa_calor(csv_path, output_dir, partida_num, fecha):
    """Genera un mapa de calor basado en los datos del CSV"""
    # 1. Leer CSV con las columnas partida, columna, fila, mina
    df = pd.read_csv(csv_path)

    # 2. Filtrar solo las posiciones con mina
    df_minas = df[df['mina'] == 1]

    # 3. Contar cuántas veces aparece una mina en cada coordenada (columna, fila)
    freq = df_minas.groupby(['columna', 'fila']).size().reset_index(name='count')

    # 4. Construir una matriz 5×5 con los conteos
    matrix = np.zeros((5, 5), dtype=int)
    for _, row in freq.iterrows():
        col_idx = int(row['columna']) - 1
        row_idx = int(row['fila'])    - 1
        matrix[row_idx, col_idx] = row['count']

    # 5. Generar el mapa de calor con anotaciones
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(matrix, cmap='Reds', aspect='equal')

    # Añadir anotaciones de los conteos en cada casilla
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j],
                    ha='center', va='center',
                    color="#e63636" if matrix[i, j] > matrix.max()/2 else "white",
                    fontsize=12, weight='bold')

    # Ajustes de ejes y colorbar
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(np.arange(1, 6))
    ax.set_yticklabels(np.arange(1, 6))
    ax.set_xlabel('Columna', fontsize=12)
    ax.set_ylabel('Fila', fontsize=12)
    ax.set_title(f'Mapa de calor de recurrencia de minas - Partida {partida_num}', fontsize=14)
    fig.colorbar(cax, label='Número de minas')

    plt.tight_layout()
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar la imagen
    filename = f"minas_partida_{partida_num}_{fecha}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def obtener_siguiente_partida(output_dir):
    """Obtiene el número de la siguiente partida basado en archivos existentes en la carpeta de la fecha actual"""
    if not os.path.exists(output_dir):
        return 1
    
    archivos = [f for f in os.listdir(output_dir) if f.startswith("minas_partida_") and f.endswith(".png")]
    if not archivos:
        return 1
    
    numeros = []
    for archivo in archivos:
        try:
            # Extraer número de partida del nombre del archivo
            parte = archivo.replace("minas_partida_", "").split("_")[0]
            numeros.append(int(parte))
        except:
            continue
    
    return max(numeros) + 1 if numeros else 1

def main():
    # Obtener fecha actual
    fecha_actual = datetime.now().strftime("%Y-%m-%d")
    
    # 1. Pedir al usuario la ruta del Excel
    excel_path = input("Introduce la ruta al archivo .xlsx con posiciones de minas: ").strip()
    if not os.path.isfile(excel_path):
        print(f"❌ Error: no se encontró el archivo '{excel_path}'")
        sys.exit(1)

    # 2. Cargar el Excel
    df_pos = pd.read_excel(excel_path, engine='openpyxl')
    required = {'partida', 'fila', 'columna'}
    if not required.issubset(df_pos.columns):
        print(f"❌ El archivo debe tener columnas: {required}")
        sys.exit(1)

    # 3. Construir lista de registros (iterando columnas primero)
    registros = []
    for partida, grupo in df_pos.groupby('partida'):
        bombas = set(zip(grupo['columna'], grupo['fila']))
        for col in range(1, 6):
            for fila in range(1, 6):
                registros.append({
                    'partida': partida,
                    'columna': col,
                    'fila':    fila,
                    'mina':    1 if (col, fila) in bombas else 0
                })

    # 4. Crear DataFrame con el orden de columnas deseado
    df_reg = pd.DataFrame(registros, columns=['partida', 'columna', 'fila', 'mina'])

    # 4.1 Verificar/crear carpeta 'games'
    games_dir = "games"
    if not os.path.exists(games_dir):
        os.makedirs(games_dir, exist_ok=True)
        print(f"✅ Carpeta '{games_dir}' creada")
    else:
        print(f"✅ Carpeta '{games_dir}' ya existe")

    # 4.2 Verificar/crear carpeta 'heat_maps_mines'
    base_output_dir = "heat_maps_mines"
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir, exist_ok=True)
        print(f"✅ Carpeta '{base_output_dir}' creada")
    else:
        print(f"✅ Carpeta '{base_output_dir}' ya existe")

    # 5. Crear nombre de archivo CSV con fecha en la carpeta 'games'
    salida = os.path.join(games_dir, f'minas-{fecha_actual}.csv')
    if os.path.exists(salida):
        df_reg.to_csv(salida, mode='a', header=False, index=False, encoding='utf-8')
        print(f"✅ Datos agregados al archivo existente '{salida}'")
    else:
        df_reg.to_csv(salida, mode='w', header=True,  index=False, encoding='utf-8')
        print(f"✅ Archivo CSV creado: '{salida}'")

    # 6. Generar mapa de calor
    graphics_dir = f"graphics_{fecha_actual}"
    output_dir = os.path.join(base_output_dir, graphics_dir)
    partida_num = obtener_siguiente_partida(output_dir)
    
    try:
        filepath_imagen = generar_mapa_calor(salida, output_dir, partida_num, fecha_actual)
        print(f"✅ Mapa de calor generado: '{filepath_imagen}'")
    except Exception as e:
        print(f"❌ Error al generar el mapa de calor: {e}")

    print(f"\n📊 Resumen:")
    print(f"   • Archivo CSV: {salida}")
    print(f"   • Mapa de calor: {os.path.join(output_dir, f'minas_partida_{partida_num}_{fecha_actual}.png')}")
    print(f"   • Carpeta de gráficos: {output_dir}")
    print(f"   • Partida número: {partida_num}")
    print("\nEl programa se cerrará en 10 segundos...")
    time.sleep(10)

if __name__ == "__main__":
    main()
