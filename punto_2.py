import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


X_ = pd.read_csv('/Users/belengotz/Desktop/dataset_x_y/dataset01.csv').values 
X = StandardScaler().fit_transform(X_)

U, S, Vt = np.linalg.svd(X, full_matrices=False)


d_values = [2, 6, 10, X.shape[1]] 
sigma = 10 #IR CAMBIANDO

plt.subplot(1, 1, 1)  # Primer gráfico en una cuadrícula de 3x2
plt.imshow(X, interpolation='nearest', aspect='auto', cmap='viridis')
plt.title('Matriz original (X)')
plt.colorbar()

def calculate_similarity(X, sigma):
    """
    Calcula la matriz de similaridad utilizando el kernel RBF
    :param X: Matriz de datos n x p (n muestras y p características)
    :param sigma: Parámetro del kernel RBF
    :return: Matriz de similaridad n x n
    """
    dist_matrix = euclidean_distances(X, X) 
    
    similarity_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    
    return similarity_matrix


V_2 = Vt[:2, :]  
Z2 = X @ V_2.T

plt.figure(figsize=(12, 8))  
plt.scatter(Z2[:, 0], Z2[:, 1], s=50, c='b', marker='o')
plt.title(f'Proyección en 2 dimensiones (d={2})')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.show()

for idx, d in enumerate(d_values, 1):
    V_d = Vt[:d, :]  
    Z = X @ V_d.T  

    similarity_reduced = calculate_similarity(Z, sigma)

    plt.subplot(2, 2, idx)  
    plt.imshow(similarity_reduced, interpolation='nearest', aspect='auto', cmap='viridis')
    plt.title(f'Similaridad en espacio reducido (d={d})')
    plt.colorbar()

plt.tight_layout()
plt.show()




"""

representacion de la matriz og
todo ruido excepto las ult 6 columnas q son cl entre ellas (2 que son las importantes)

al poner la imagen de la matriz, decir eso de antes

para poder analizar los features del espacio og mas importantes


Y CORROBORAR ESO CON PCA (FEATURES QUE MAYOR VARIANZA EXPRESA)

preguntarle a fran lo de recien para lo anterior!!!



HACER CUADRADOS MINIMOS CON LA PSEUDO INVERSA (no usar la funcion least squares)
beta es hacer la pseudo inversa por y




"""

# Paso 1: Cargar el vector de etiquetas Y desde el archivo y.txt y restarle su media
Y_ = pd.read_csv('/Users/belengotz/Desktop/dataset_x_y/y1.txt', header=None).values.flatten()
Y_mean = np.mean(Y_)
Y = Y_ - Y_mean  # Restarle la media a Y

Y_centered = Y - np.mean(Y)

# Paso 1: Calcular la pseudo-inversa de X
# La pseudo-inversa es (X^T X)^{-1} X^T
X_pseudo_inverse = np.linalg.inv(X.T @ X) @ X.T

# Paso 2: Calcular el vector B usando la pseudo-inversa
B = X_pseudo_inverse @ Y_centered

# Mostrar el vector B
print("Vector B que minimiza el error de predicción:")
print(B)

# Mostrar el error cuadrático medio del modelo
mse = np.mean((X @ B - Y_centered) ** 2)
print(f"Error cuadrático medio del modelo: {mse}")

# Visualización de los pesos asignados a cada dimensión original
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(B) + 1), B)
plt.xlabel("Dimensión original")
plt.ylabel("Peso asignado (B)")
plt.title("Pesos asignados a cada dimensión original en el modelo de cuadrados mínimos")
plt.show()


"""
Los pesos más altos (positivos o negativos) indican que las correspondientes dimensiones originales de 
X
X tienen una mayor importancia en la predicción de 
Y
Y, mientras que los pesos cercanos a cero sugieren que esas dimensiones contribuyen poco al modelo.
Este análisis puede ayudar a identificar cuáles características son las más relevantes y pueden guiar la selección de variables, especialmente en modelos de alta dimensionalidad.
"""
