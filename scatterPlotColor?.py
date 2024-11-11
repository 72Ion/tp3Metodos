import matplotlib.pyplot as plt
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


V_2 = Vt[:2, :]  
Z2 = X @ V_2.T

plt.figure(figsize=(14, 10))

# Fondo de color claro para hacerlo más "chato"
plt.gca().set_facecolor('#F5F5F5')  

# Gráfico de dispersión con una paleta en tonos pastel y opacidad ajustada
scatter = plt.scatter(
    Z2[:, 0], Z2[:, 1],
    s=60,                          # Tamaño de los puntos
    c=np.linalg.norm(Z2, axis=1),  # Color basado en la magnitud de Z2
    cmap='viridis',               # Paleta de colores suave
    marker='o',
    edgecolor='w',                 # Bordes blancos para resaltar puntos
    alpha=0.85                     # Transparencia para suavizar el aspecto
)

# Barras de color
colorbar = plt.colorbar(scatter, shrink=0.8, aspect=10, pad=0.02)
colorbar.set_label('Intensidad de Magnitud', fontsize=12, color='#333333')
colorbar.ax.tick_params(colors='#555555')

# Título y etiquetas de ejes con fuente y color personalizadas
plt.title(f'Proyección en 2 Dimensiones (d={2})', fontsize=16, fontweight='bold', color='#333333')
plt.xlabel('Componente 1', fontsize=14, color='#555555')
plt.ylabel('Componente 2', fontsize=14, color='#555555')

# Ajuste del color y tamaño de las marcas de los ejes
plt.xticks(fontsize=12, color='#555555')
plt.yticks(fontsize=12, color='#555555')

# Límites de los ejes para un aspecto más simétrico y cuadrado
plt.xlim(Z2[:, 0].min() - 1, Z2[:, 0].max() + 1)
plt.ylim(Z2[:, 1].min() - 1, Z2[:, 1].max() + 1)

plt.grid(color='#e0e0e0', linestyle='--', linewidth=0.5)  # Cuadrícula sutil en gris claro

plt.show()