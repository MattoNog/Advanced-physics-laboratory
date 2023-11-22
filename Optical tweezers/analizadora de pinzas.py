# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 00:13:19 2023

@author: matia
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import csv


#%% dataframe
df = pd.read_csv('C:/Users/matia/OneDrive/Escritorio/prueba 01')
t = df["t"]
column_prefix = "r_medio_"


numero_de_columnas = df.shape[1] 
num_columns = int(3**(-1) * (numero_de_columnas -1))


dataframe = np.zeros((num_columns,len(t)))
err_dataframe = np.zeros((num_columns,len(t)))
rtos_ajuste = np.zeros((num_columns,5))

for i in range(num_columns):
    column_name = f"{column_prefix}{i}"
    dataframe[i,:] = df[column_name]
    
    
#%% graficadora

colores = ["#FF5733", "#33FF57", "#3366FF", "#FF33C9", "#FFD933", "#33FFD9", "#FF3366", "#33D9FF", "#9933FF", "#33FFC9", "#FF3333", "#D9FF33", "#3366D9", "#FF5733", "#33FF57"]

for i in range(num_columns):
    plt.plot(t,dataframe[i,:], label = f"partícula {i}", color= colores[i] )
plt.xlabel('t')
plt.ylabel('r')
plt.legend()
plt.grid(linestyle='--')
plt.show()


#%% Automatizo el analísis
def func(m, A, B):
  return A*m + B

for i in range(num_columns): 
    init_guess = [1, 0.01]
    param, param_cov = curve_fit(func, t, dataframe[0,:], p0=init_guess)
    
    print(param)
    print(param_cov)
    
    perr = np.sqrt(np.diag(param_cov)) #la variación en cada parámetro
    
    xdata = t      
    ydata = dataframe[i,:] #INPUT
    yerr_0 = err_dataframe[i,:]      #INPUT
    
    puntos = len(xdata)
    parametros_chi = len(param)
    y_modelo = func(xdata, param[0], param[1])
    y = xdata
    yerr = yerr_0
    
    chi_cuadrado = np.sum(((y-y_modelo)/yerr)**2)
    p_chi = stats.chi2.sf(chi_cuadrado, puntos - 1 - parametros_chi)
    chi_cuadrado_redux = chi_cuadrado/4
    
    print('chi^2: ' + str(chi_cuadrado))
    print('p-valor del chi^2: ' + str(p_chi))
    print('chi^2 reducido: ' + str(chi_cuadrado_redux))
    print("Pendiente =", param[0])
    print("Ordenada =", param[1])
    
    #plot del ajuste
    
    x = np.linspace(min(xdata), max(xdata), 1000)
    plt.figure(figsize=(15,4))
    plt.errorbar(xdata, ydata,  c='purple', label = "Datos", yerr = yerr_0, fmt='o',  ecolor='grey', capsize=2, markersize = 7, elinewidth = 2,)
    plt.plot(x, func(x, param[0], param[1]), label = 'Ajuste')
    plt.ylabel(r'Tau Potencial', fontsize=15) 
    plt.xlabel(r'Resistencia', fontsize=15)
    plt.legend(fontsize = 13)
    plt.grid(linestyle='--')
    plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )
    plt.plot
    
    #plot de residuos
    
    plt.figure(figsize=(15,4))
    x = np.linspace(min(xdata), max(xdata), 1000)
    plt.errorbar(xdata, ydata-func(xdata, param[0], param[1]), fmt='o', markersize = 7, label = 'Residuo')
    plt.axhline(y = 0, color = 'r', linestyle = '--', label = 'y=0')
    plt.ylabel(r'$<r^{2}>$', fontsize=15)
    plt.xlabel(r'$Tiempo$', fontsize=15)
    plt.xticks(np.arange(min(xdata), max(xdata), 1000))
    plt.minorticks_on()
    plt.grid(linestyle='--')
    plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )
    plt.legend(fontsize = 13)
    plt.show()

    rtos_ajuste[i,0] = param[0]
    rtos_ajuste[i,1] = param[1]
    rtos_ajuste[i,2] = p_chi
    rtos_ajuste[i,3] = chi_cuadrado_redux
    rtos_ajuste[i,4] = chi_cuadrado

# Define los nombres de las columnas
nombres_columnas = ['m', 'v0', 'p-valor del chi^2', 'chi^2 reducido', 'chi^2']

# Nombre de archivo CSV
nombre_archivo = 'RTOS_AJUSTE.csv'

# Abre el archivo CSV en modo escritura
with open(nombre_archivo, 'w', newline='') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv)

    # Escribe los nombres de las columnas como la primera fila
    escritor_csv.writerow(nombres_columnas)

    # Escribe cada fila de la matriz en el archivo CSV
    for fila in rtos_ajuste:
        escritor_csv.writerow(fila)

