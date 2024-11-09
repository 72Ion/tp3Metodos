import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.preprocessing import StandardScaler

# Cargar el dataset desde el archivo CSV
X = pd.read_csv('/Users/belengotz/Desktop/dataset_x_y/dataset01.csv').values  # Convierte el DataFrame a una matriz numpy
X = StandardScaler().fit_transform(X)

# Paso 1: Realizar la descomposición en valores singulares (SVD)
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Paso 2: Proyectar los datos al nuevo espacio reducido Z para diferentes valores de d
d_values = [2, 6, 10, X.shape[1]]  # valores de d = 2, 6, 10 y p (dimensionalidad original)

# Definir el parámetro sigma para la similitud
sigma = 15  # Reducir el valor de sigma puede ayudar a que las distancias sean más diferenciadas

def calculate_similarity(X, sigma=0.5):
    """
    Calcula la matriz de similaridad utilizando el kernel RBF
    :param X: Matriz de datos n x p (n muestras y p características)
    :param sigma: Parámetro del kernel RBF
    :return: Matriz de similaridad n x n
    """
    # Calcular la matriz de distancias euclidianas entre todas las muestras
    dist_matrix = euclidean_distances(X, X)  # dist_matrix[i, j] = ||x_i - x_j||
    
    # Aplicar el kernel RBF para convertir distancias a similaridades
    similarity_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    
    return similarity_matrix

# Calcular y mostrar la similaridad en el espacio original y reducido, y aplicar la regresión ridge
for d in d_values:
    # Reducir la dimensionalidad utilizando los primeros d vectores singulares
    V_d = Vt[:d, :]  # Tomar los primeros d vectores de Vt
    Z = X @ V_d.T  # Proyección de X en el nuevo espacio reducido
    
    # Visualización si d = 2 (para 2 dimensiones, podemos graficar)
    if d == 2:
        plt.figure(figsize=(12, 8))  # Tamaño más grande de la figura
        plt.scatter(Z[:, 0], Z[:, 1], s=50, c='b', marker='o')
        plt.title(f'Proyección en 2 dimensiones (d={d})')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        plt.show()

    # Mostrar la forma del espacio proyectado Z
    print(f"Forma del espacio proyectado Z para d={d}: {Z.shape}")
    
    # Calcular la matriz de similaridad para el espacio original X
    similarity_original = calculate_similarity(X)
    # Calcular la matriz de similaridad para el espacio reducido Z
    similarity_reduced = calculate_similarity(Z)
    
    # Visualizar las matrices de similaridad con escalas de color distintas
    plt.figure(figsize=(16, 8))  # Aumentar el tamaño de la figura

    # Similaridad en el espacio original
    plt.subplot(1, 2, 1)
    plt.imshow(similarity_original, interpolation='nearest', aspect='auto')
    plt.title(f'Similaridad en el espacio original (d={d})')
    plt.colorbar()

    # Similaridad en el espacio reducido
    plt.subplot(1, 2, 2)
    plt.imshow(similarity_reduced, interpolation='nearest', aspect='auto')
    plt.title(f'Similaridad en el espacio reducido (d={d})')
    plt.colorbar()

    plt.tight_layout()  # Para evitar superposiciones
    plt.show()

    # Imprimir la comparación de matrices de similaridad
    print(f"Comparación de similaridad: Espacio original vs Espacio reducido para d={d}")
    print("Matriz de similaridad original:\n", similarity_original)
    print("Matriz de similaridad reducida:\n", similarity_reduced)

    # Reflexión sobre la elección de 'd'
    print(f"Reflexión sobre la elección de d={d}:")
    if d == X.shape[1]:
        print("Cuando d es igual a la dimensionalidad original, la similaridad se conserva exactamente.")
    else:
        print("Al reducir la dimensionalidad, algunas relaciones entre las muestras pueden perderse, dependiendo de cuán importante es la varianza explicada por los primeros componentes.")
