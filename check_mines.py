import pandas as pd
import numpy as np

# Cargar todos los archivos
all_data = []
for f in ['games/minas-2025-06-28.csv', 'games/minas-2025-07-04.csv', 'games/minas-2025-08-06.csv', 'games/minas-2025-08-07.csv']:
    df = pd.read_csv(f)
    all_data.append(df)

combined = pd.concat(all_data)
mines_per_game = combined.groupby('partida')['mina'].sum()

print('Distribución completa de minas por partida:')
print(mines_per_game.value_counts().sort_index())
print(f'Total partidas: {len(mines_per_game)}')
print(f'Minas predominantes: {mines_per_game.mode()[0]}')

# Solo partidas con 3 minas
games_with_3_mines = mines_per_game[mines_per_game == 3]
print(f'Partidas con exactamente 3 minas: {len(games_with_3_mines)}')

# Solo partidas con 4 minas  
games_with_4_mines = mines_per_game[mines_per_game == 4]
print(f'Partidas con exactamente 4 minas: {len(games_with_4_mines)}')
