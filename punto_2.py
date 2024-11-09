import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar el dataset (ejemplo desde un archivo CSV)
def load_dataset(filepath):
    data = pd.read_csv(filepath)
    X = data.values  # Convertir el dataset en una matriz numpy
    return X

# Calcular la matriz de similitud usando la función de similitud dada
def similarity_matrix(X, sigma=1.0):
    n = X.shape[0]
    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_sq = np.sum((X[i] - X[j]) ** 2)
            similarity[i, j] = np.exp(-dist_sq / (2 * sigma ** 2))
    return similarity

# Realizar SVD y proyectar a espacio reducido de dimensión d
def svd_projection(X, d):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Proyección a espacio reducido de dimensión d
    X_reduced = U[:, :d] @ np.diag(S[:d])
    return X_reduced

# Visualizar la matriz de similitud
def plot_similarity_matrix(similarity, title):
    plt.imshow(similarity, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Muestra')
    plt.ylabel('Muestra')
    plt.show()

# Cargar el dataset
dataset_path = '/ruta/al/dataset.csv'  # Cambia esta ruta por la de tu archivo
X = load_dataset(dataset_path)

# Valores de d para probar
d_values = [2, 6, 10]
sigma = 1.0  # Parámetro de similitud

# Calcular y visualizar la matriz de similitud en el espacio original
similarity_X = similarity_matrix(X, sigma=sigma)
plot_similarity_matrix(similarity_X, 'Similitud en el Espacio Original')

# Calcular y visualizar la matriz de similitud en el espacio reducido para cada valor de d
for d in d_values:
    X_reduced = svd_projection(X, d)
    similarity_Z = similarity_matrix(X_reduced, sigma=sigma)
    plot_similarity_matrix(similarity_Z, f'Similitud en el Espacio Reducido (d={d})')

    # Comparar similitudes: calcular diferencia entre matrices de similitud
    diff = np.linalg.norm(similarity_X - similarity_Z, 'fro') / np.linalg.norm(similarity_X, 'fro')
    print(f"Diferencia relativa de similitud para d = {d}: {diff:.4f}")
