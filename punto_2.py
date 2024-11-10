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


sigma = 5 #IR CAMBIANDO

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