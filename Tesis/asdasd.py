import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.integrate import quad
from astroquery.ned import Ned

galaxy = "ic342"
carpeta = "./galaxias/"+ galaxy +"_es3_filter"
###########

# leer los datos con pandas
df = pd.read_csv(carpeta + ".txt" , header=0, delimiter='\s+')

# Obtiene informacion de la base de datos NED
result_table = Ned.query_object(galaxy)
redshift = result_table['Redshift'][0]
# Imprime el redshift
print(f"Redshift: {redshift}")

#Correccion por redshift 
df_corr = df.copy()  # Crear una copia del DataFrame original
df_corr['frec(GHz)'] = round((df_corr['frec(GHz)']) *(redshift+1),3)

# Lee los datos de las frecuencias
tab_frec = pd.read_csv("./galaxias/frecuencias.txt" , header=0, delimiter='\s+')
tab_frec['Rest_Freq'] = round(tab_frec['Rest_Freq'],3 )


# Definir el nombre de la molécula de interés
molecule_name = "C2H" #C2H

# Buscar la fila correspondiente a la molécula de interés y obtener el valor de Rest_Freq
rest_freq = tab_frec.loc[tab_frec["Molecule"] == molecule_name, "Rest_Freq"].values[0]

# Se ponen 3 puntos antes y despues del dato en la tabla de frecuencias
lim_inf, lim_sup = rest_freq - 0.1 , rest_freq + 0.167

# enmascarar valores
mask = (df_corr['frec(GHz)'] >= lim_inf) & (df_corr['frec(GHz)'] <= lim_sup)
masked_data = df_corr.loc[mask]

# eliminar filas con valores no finitos
masked_data = masked_data[np.isfinite(masked_data['Temperatura[mK]'])]
##################


def Cubic_Spline(df_spline,lim_inf, lim_sup):
    # realiza la interpolación cúbica
    adjSpline = CubicSpline(df_spline['frec(GHz)'], df_spline['Temperatura[mK]'],bc_type='natural')
    x_interpolation = np.linspace(lim_inf, lim_sup)
    return x_interpolation, adjSpline

x_CubicSpline, y_CubicSpline = Cubic_Spline(masked_data,lim_inf, lim_sup)
y_CubicSpline = y_CubicSpline(x_CubicSpline)





# Calcula los parámetros para una o N-gaussianas
def ngaussians(x, *params):
    n = len(params) // 3
    y = np.zeros_like(x)
    for i in range(n):
        amp, mu, sigma = params[i*3:(i+1)*3]
        y += amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y

# Permite seleccionar puntos en el ploteo de los datos 
def seleccionar_puntos(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    puntos_seleccionados = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            distancia_minima, punto_seleccionado = float('inf'), None

            # Selecciona el punto más cercano calculando la hipotenusa y lo compara con los otros resultados.
            for i in range(len(x)):
                distancia = np.sqrt((event.xdata - x[i])**2 + (event.ydata - y[i])**2)
                if distancia < distancia_minima:
                    distancia_minima = distancia
                    punto_seleccionado = (x[i], y[i])

            puntos_seleccionados.append(punto_seleccionado)
        print(f"Puntos seleccionados: {puntos_seleccionados}")

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return puntos_seleccionados

x = [1.321, 2.332, 3.422, 4.323, 5.323]
y = [2.322, 4.434, 6.523, 8.234, 10.643]

x1 = [87.22, 87.22544898, 87.23089796, 87.23634694, 87.24179592, 87.2472449, 87.25269388, 87.25814286, 87.26359184, 87.26904082, 87.2744898, 87.27993878, 87.28538776, 87.29083673, 87.29628571, 87.30173469, 87.30718367, 87.31263265, 87.31808163, 87.32353061, 87.32897959, 87.33442857, 87.33987755, 87.34532653, 87.35077551, 87.35622449, 87.36167347, 87.36712245, 87.37257143, 87.37802041, 87.38346939, 87.38891837, 87.39436735, 87.39981633, 87.40526531, 87.41071429, 87.41616327, 87.42161224, 87.42706122, 87.4325102, 87.43795918, 87.44340816, 87.44885714, 87.45430612, 87.4597551, 87.46520408, 87.47065306, 87.47610204, 87.48155102, 87.487]

y2= [1.21423772, 0.60988314, -0.03005671, -0.6295717, -1.11265173, -1.40328669, -1.42546647, -1.10318096, -0.36840221, 0.75153657, 2.16653684, 3.78539669, 5.51691425, 7.26988761, 8.94451937, 10.40339452, 11.49881008, 12.08306286, 12.00844963, 11.12906843, 9.46260451, 7.31874714, 5.03740553, 2.95848889, 1.4219064, 0.74670542, 0.92935965, 1.71258625, 2.83225284, 4.02422705, 5.02437649, 5.5913124, 5.70340907, 5.46215698, 4.97040223, 4.33099095, 3.64676882, 3.00332309, 2.42313609, 1.9142708, 1.48479021, 1.14275732, 0.89606861, 0.74314954, 0.66803595, 0.65355919, 0.6825506, 0.73784156, 0.8022634, 0.85864749]
data = [[x,y],[x1,y2]]


for x_list,y_list in data:
    puntos_seleccionados = sorted(seleccionar_puntos(x_list, y_list), key=lambda x: x[0])
    print(puntos_seleccionados)
    
    x_list, y_list = np.array(x_list), np.array(y_list)

    if len(puntos_seleccionados)>=2:
        x_mask = x_list[(x_list >= puntos_seleccionados[0][0]) & (x_list <= puntos_seleccionados[1][0])]
        y_mask = y_list[(x_list >= puntos_seleccionados[0][0]) & (x_list <= puntos_seleccionados[1][0])]
        
        a_o = max(y_mask)  # Amplitud estimada como el valor máximo de y
        mu_o = x_mask[np.argmax(y_mask)]  # Media estimada como el valor de X correspondiente al máximo de y
        sigma_o = (max(x_mask) - min(x_mask)) / 4 

        p0=[a_o,mu_o,sigma_o]
        p0=[ 5, 87.4, 0.1]
        params, covariance = curve_fit(ngaussians, x_CubicSpline, y_CubicSpline, p0=p0)

        perr = np.sqrt(np.diag(covariance))

        residual = y_CubicSpline - (ngaussians(x_CubicSpline, *params))

        fig = plt.figure()
        gs = gridspec.GridSpec(2,1, height_ratios=[1,0.30])
        ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        
        gs.update(hspace=0) 

        ax1.plot(x_CubicSpline, y_CubicSpline, "ro",label="Datos")
        ax1.plot(x_CubicSpline, ngaussians(x_CubicSpline, *params), 'k--',label="Ajuste")
        for i in range(int(len(params)/3)):
            ax1.plot(x_CubicSpline, ngaussians(x_CubicSpline, *params[i*3:(i+1)*3]), label= "Gaussiana " + str(i+1))
        ax2.plot(x_CubicSpline, residual, "bo") # reciduo

        ax1.set_title(galaxy)
        ax2.set_xlabel('Frecuencia (GHz)')
        ax1.set_ylabel('Ta (mK)')
        ax2.set_ylabel("Res.")
        ax1.legend()

        fig.savefig("fitGaussian.png", format="png",dpi=1000)

        # Iterar sobre cada curva para imprimir los parámetros de ajuste y su error
        for i in range(int(len(params)/3)):
            offset = i * 3  # para obtener los índices correctos de los parámetros en params
            print("-------------Curva {}-------------".format(i+1))
            print("Amplitud (a) = %0.3f (+/-) %0.3f" % (params[offset], perr[offset]))
            print("Media (mu) = %0.3f (+/-) %0.3f" % (params[offset+1], perr[offset+1]))
            print("Desviación estándar (sigma) = %0.3f (+/-) %0.3f" % (params[offset+2], perr[offset+2]))
        print()

        # Cálculo de la integral de la gaussiana ajustada
        def integral(x_CubicSpline):
            return ngaussians(x_CubicSpline, *params)

        integral, error = quad(integral, min(x_CubicSpline), max(x_CubicSpline))
        print("Area total = %0.3f (+/-) %0.3f" % (integral, error))