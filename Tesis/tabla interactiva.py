import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Trace su gráfica
ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'o')

selected_points = []

def onpick(event):
    # Obtenga las coordenadas del punto seleccionado
    x, y = event.xdata, event.ydata
    selected_points.append((x, y))
    print(f'Punto seleccionado: x={x}, y={y}')
    
# Active la herramienta "Seleccionar puntos" en su gráfica
fig.canvas.mpl_connect('pick_event', onpick)

plt.show()

print('Coordenadas seleccionadas:', selected_points)
