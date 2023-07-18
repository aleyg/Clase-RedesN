import pandas as pd 
tab_frec = pd.read_csv("./galaxias/frecuencias.txt" , header=0, delimiter='\s+')
tab_frec

import plotly.graph_objects as go

# Crear una tabla interactiva con Plotly
fig = go.Figure(data=[go.Table(
    header=dict(values=list(tab_frec.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[tab_frec.Molecule, tab_frec.Transition, tab_frec.Rest_Freq],
               fill_color='lavender',
               align='left'))
])
# Agregar una función para seleccionar datos con clicks
def update_selected_cells(trace, points, state):
    indices = [p['pointIndex'] for p in points]
    selected_data = tab_frec.iloc[indices]
    print(selected_data)

fig.data[0].on_click(update_selected_cells)

# Mostrar la tabla interactiva
fig.show()