import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
1. Para hacer esto hay que realizar una descomposición de X en sus valores singulares, 
reducir la dimensión de esta representación, y luego trabajar con los vectores x proyectados 
al nuevo espacio reducido Z,
es decir z = Vx. Realizar los puntos anteriores para d = 2, 6, 10, y p.
"""

X_ = pd.read_csv('/Users/belengotz/Desktop/dataset_x_y/dataset01.csv')
X_ = X_.drop(X_.columns[0], axis = 1) #ya lo lee
X = StandardScaler().fit_transform(X_)

#histograma a cada columna
#"""
num_columns = X.shape[1]
selected_columns = list(range(6)) + list(range(num_columns - 6, num_columns))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, i in enumerate(selected_columns[:6]):
    ax = axes[idx // 3, idx % 3]
    ax.hist(X[:, i], bins=30, color='blue', alpha=0.7)
    ax.set_title(f'Histograma de la columna {i+1}')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frecuencia')

plt.tight_layout()
plt.suptitle('Histogramas de las primeras 6 columnas', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()

# Plotear los últimos 6 histogramas (3 y 3) en otra imagen
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, i in enumerate(selected_columns[6:]):
    ax = axes[idx // 3, idx % 3]
    ax.hist(X[:, i], bins=30, color='green', alpha=0.7)
    ax.set_title(f'Histograma de la columna {i+1}')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frecuencia')

plt.tight_layout()
plt.suptitle('Histogramas de las últimas 6 columnas', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()
#""

""" 1 """
Y_ = np.loadtxt('/Users/belengotz/Desktop/dataset_x_y/y1.txt').reshape(-1, 1)  # Cargar y
scaler_Y = StandardScaler()
Y_standardized = scaler_Y.fit_transform(Y_).flatten()  # Normalizar Y
#print(X)

U, S, Vt = np.linalg.svd(X, full_matrices=False)

#VISUALIZAR A
plt.subplot(1, 1, 1)
plt.imshow(X, interpolation='nearest', aspect='auto', cmap='viridis')
plt.title('Matriz original (X)')
plt.colorbar()

#SCATTERPLOT 2D --> reducción de dimensiones
V_2 = Vt[:2, :]  
Z2 = X @ V_2.T
plt.figure(figsize=(12, 8))
scatter = plt.scatter(Z2[:, 0], Z2[:, 1], s=50, c=Y_, cmap='viridis', marker='o')
plt.colorbar(scatter, label='Valor de Y estandarizado')  # Barra de color que indica los valores de Y
plt.title(f'Proyección en 2 dimensiones (d=2) con color basado en Y')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
#plt.show()

"""
2. Analizar la similaridad par-a-par entre muestras en el espacio de dimension X y en el espacio de dimensión reducida d para distintos valores de d utilizando PCA. Comparar estas medidas de similaridad
Ayuda: ver de utilizar una matriz de similaridad para visualizar todas las similaridades par-a-par juntas.
¿Para qué elección de d resulta más conveniente hacer el análisis? ¿Cómo se conecta esto con los valores singulares de X? ¿Qué conclusiones puede sacar al respecto?
"""
def calculate_similarity(X, sigma): #similaridad par a par
    """
    Calcula la matriz de similaridad utilizando el kernel RBF
    :param X: Matriz de datos n x p (n muestras y p características)
    :param sigma: Parámetro del kernel RBF
    :return: Matriz de similaridad n x n
    """
    dist_matrix = euclidean_distances(X, X) 
    
    similarity_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    
    return similarity_matrix

#PARÁMETROS A EVALUAR
d_values = [2, 6, 10, X.shape[1]] 
#sigma = [1, 5, 10, 50, 100, 1000] #IR CAMBIANDO
sigma = [10000];
for s in sigma:
    for idx, d in enumerate(d_values, 1):
        V_d = Vt[:d, :]  
        Z = X @ V_d.T  

        similarity_reduced = calculate_similarity(Z, s)

        plt.subplot(2, 2, idx)  
        plt.imshow(similarity_reduced, interpolation='nearest', aspect='auto', cmap='viridis')
        plt.title(f'Similaridad en espacio reducido (d={d})')
        plt.colorbar()

    plt.tight_layout()
    plt.suptitle(f'Similaridad en espacio reducido con σ ={s}')
    #plt.show()

"""
for d in d_values:
    V_d = Vt[:d, :]  # Selección de las primeras d componentes principales
    Z = X @ V_d.T  # Proyección de X en el espacio reducido de d dimensiones
    
    optimal_sigma_value = calculate_optimal_sigma(X)
    similarity_reduced = calculate_similarity(Z, optimal_sigma_value)

    plt.subplot(2, 2, d)  
    plt.imshow(similarity_reduced, interpolation='nearest', aspect='auto', cmap='viridis')
    plt.title(f'Similaridad en espacio reducido (d={d})')
    plt.colorbar()

    plt.tight_layout()
    plt.suptitle(f'Similaridad en espacio reducido con σ ={optimal_sigma_value}')
    plt.show()
        

    print(f"Para d={d}, valor óptimo de sigma (calculado) = {optimal_sigma_value}")

"""    
# Los pesos más altos (positivos o negativos) indican que las correspondientes dimensiones originales de X tienen una mayor importancia en la predicción de Y, 
# mientras que los pesos cercanos a cero sugieren que esas dimensiones contribuyen poco al modelo.
# Este análisis puede ayudar a identificar cuáles características son las más relevantes y pueden guiar la selección de variables, 
# especialmente en modelos de alta dimensionalidad. ??????????????

""" 
3. Los datos X vienen acompañados de una variable dependiente respuesta o etiquetas llamada Y (archivo y.txt) estructurada como un vector n x 1 para cada muestra.
Queremos encontrar el vector ® y modelar linealmente el problema que minimice la norma. |XB -yll2
de manera tal de poder predecir con XB - ý lo mejor posible a las etiquetas y, es decir, 
minimizar el error de predicción utilizando todas las variables iniciales.
Resolviendo el problema de cuadrados mínimos en el espacio original X, 
que peso se le asigna a cada dimensión original si observamos el vector B?

4. Usando la representacion aprendida con PCA y d - 2: mejora la predicción || ZB - y lle en comparacion a no realizar reduccion de dimensionalidad? 
Cuales muestras son las de mejor predicción con el mejor modelo?
"""

"""OLS usando todo el conjunto de datos"""
X_pseudo_inverse = np.linalg.pinv(X)  # pseudo-inversa de X
B = X_pseudo_inverse @ Y_standardized  # estimación de los coeficientes B
Y_pred_all = X @ B

#(ECM) en todo el conjunto
mse_all_original = mean_squared_error(Y_standardized, Y_pred_all)
print(f"Error cuadrático medio (ECM) con todo el conjunto: {mse_all_original}")

B_list = B.flatten().tolist()
print("Pesos de los coeficientes B:", B_list)

#pesos de los coeficientes B
plt.figure(figsize=(12, 6))
plt.bar(range(len(B)), B, color='purple', alpha=0.7)
plt.xlabel('Índice de las dimensiones de X')
plt.ylabel('Peso de B (Coeficientes de regresión)')
plt.title('Distribución de los pesos de los coeficientes de regresión (B) en el espacio original')
plt.grid(axis='y', linestyle='--', alpha=0.6)
#plt.show()

""" 4 """
U, S, Vt = np.linalg.svd(X, full_matrices=False)
V_2 = Vt[:2, :] 
Z = X @ V_2.T

#estimar B en el espacio reducido (2 dimensiones)
Z_pseudo_inverse = np.linalg.pinv(Z)
B_Z2 = Z_pseudo_inverse @ Y_standardized #estimación de B en el espacio reducido

Y_pred_Z2 = Z @ B_Z2 #predicción en el espacio reducido


plt.figure(figsize=(10, 6))
plt.scatter(range(len(Y_standardized)), Y_standardized, color='blue', label='Datos observados', alpha=0.5)
plt.plot(range(len(Y_standardized)), Y_pred_all, color='red', label='Predicción - Espacio Original', linewidth=2)
plt.plot(range(len(Y_standardized)), Y_pred_Z2, color='green', label='Predicción - Espacio Reducido (d=2)', linewidth=2)
plt.title('Ajuste de regresión lineal en diferentes espacios', fontsize=16)
plt.xlabel('Índice de muestra', fontsize=12)
plt.ylabel('Valor de etiqueta (y)', fontsize=12)
plt.legend()
plt.show()


# (ECM) en el espacio reducido (2 dimensiones)
mse_Z2 = mean_squared_error(Y_standardized, Y_pred_Z2)
print(f"Error cuadrático medio en el conjunto completo en el espacio reducido (d=2): {mse_Z2}")

#ver si mejora la predicción en el espacio reducido
print(f"Diferencia en el ECM con reducción de dimensionalidad: {mse_Z2 - mse_all_original}")



#diferentes 80/20 aleatorios y calcular ECM
mse_test_d = []
for i in range(1000):  # Probar 10 divisiones aleatorias
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_standardized, test_size=0.2, random_state=i)
    
    # PCA: Realizar la descomposición SVD para X_train
    U_train, S_train, Vt_train = np.linalg.svd(X_train, full_matrices=False)
    V_2_train = Vt_train[:2, :]
    Z_train = X_train @ V_2_train.T
    Z_test = X_test @ V_2_train.T

    # Estimar B en el espacio reducido (2 dimensiones)
    Z_train_pseudo_inverse = np.linalg.pinv(Z_train)
    B_Z2 = Z_train_pseudo_inverse @ Y_train

    # Predicción en el espacio reducido
    Y_pred_test_Z2 = Z_test @ B_Z2

    # Calcular el ECM y almacenar
    mse_test_Z2 = mean_squared_error(Y_test, Y_pred_test_Z2)
    mse_test_d.append(mse_test_Z2)

# Promediar los errores para ver cómo cambia el ECM con diferentes particiones
mean_mse_test_d = np.mean(mse_test_d)
print(f"Promedio de ECM para 1000 divisiones aleatorias: {mean_mse_test_d}")

plt.figure(figsize=(10, 6))
plt.bar(['Espacio Original', 'Espacio Reducido (d=2)', '1000 divisiones aleatorias'], [mse_all_original, mse_Z2, mean_mse_test_d], color=['blue', 'green'], alpha=0.7)
plt.xlabel('Espacio')
plt.ylabel('Error Cuadrático Medio (ECM)')
plt.title('Comparación de ECM en el Espacio Original y Reducido')
plt.grid(axis='y', linestyle='--', alpha=0.6)
#plt.show()

"""

Error cuadrático medio (ECM) con todo el conjunto: 0.048788277812862
Error cuadrático medio en el conjunto completo en el espacio reducido (d=2): 0.054786176945249454
Promedio de ECM para 10 divisiones aleatorias: 0.06297652578613722



HACER GRAFICO COMPARAR ERROR D2 Y TODOS
E ECM CADA UNO GRAFICOS

"""




#error abs de predicción para cada muestra
errors_Z2 = np.abs(Y_standardized - Y_pred_Z2)
best_predictions_indices = np.argsort(errors_Z2)[:10] #muestras con el error más bajo (las de mejor predicción)
print("Índices de las muestras con los mejores errores de predicción:")
print(best_predictions_indices)

errors_all = np.abs(Y_standardized - Y_pred_all)
plt.figure(figsize=(12, 6))
plt.bar(range(len(errors_Z2)), errors_Z2.flatten(), color='#DDA0DD', alpha=0.7, label='Espacio Reducido (d=2)')
plt.xlabel('Índice de muestra')
plt.ylabel('Error absoluto')
plt.title('Errores absolutos de la predicción en el espacio reducido (d=2)')
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(range(len(errors_all)), errors_all.flatten(), color='#0000FF', alpha=0.7, label='Espacio Original')
plt.xlabel('Índice de muestra')
plt.ylabel('Error absoluto')
plt.title('Errores absolutos de la predicción')
plt.show()


#visualizar estas muestras con los mejores errores en el gráfico de dispersión
plt.figure(figsize=(10, 6))
plt.scatter(Z[:, 0], Z[:, 1], c=errors_Z2, cmap='viridis', s=50, label="Muestras")
plt.scatter(Z[best_predictions_indices, 0], Z[best_predictions_indices, 1], c='red', s=100, marker='X', label="Mejores predicciones") #resaltar las muestras con los mejores errores
plt.colorbar(label="Error de predicción")
plt.title("Distribución de las muestras y mejores predicciones (errores bajos) en el espacio reducido (d=2)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend()
plt.show()

#mostrar las predicciones y los valores reales para las mejores predicciones
print("Predicciones y valores reales para las mejores predicciones:")
for idx in best_predictions_indices:
    print(f"Muestra {idx}: Predicción = {Y_pred_Z2[idx]}, Real = {Y_standardized[idx]}")





""" d optimo """

mse_values = []

# Evaluar el ECM para diferentes valores de d
for d in d_values:
    V_d = Vt[:d, :]  # Seleccionar las primeras d componentes principales
    Z = X @ V_d.T  # Proyectar los datos en el espacio reducido
    
    # Estimar los coeficientes B en el espacio reducido
    Z_pseudo_inverse = np.linalg.pinv(Z)
    B_Z = Z_pseudo_inverse @ Y_standardized
    
    # Realizar la predicción en el espacio reducido
    Y_pred_Z = Z @ B_Z
    
    # Calcular el ECM para el modelo en el espacio reducido
    mse_d = mean_squared_error(Y_standardized, Y_pred_Z)
    mse_values.append(mse_d)

optimal_d = d_values[np.argmin(mse_values)]
print(f"El valor óptimo de d es {optimal_d} con un ECM de {min(mse_values)}")    
