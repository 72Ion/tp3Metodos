import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

"""
1. Para hacer esto hay que realizar una descomposición de X en sus valores singulares, reducir la dimensión de esta representación, y luego trabajar con los vectores x proyectados al nuevo espacio reducido Z,
es decir z = Vx. Realizar los puntos anteriores para d = 2, 6, 10, y p.

2. Analizar la similaridad par-a-par entre muestras en el espacio de dimension X y en el espacio de dimensión reducida d para distintos valores de d utilizando PCA. Comparar estas medidas de similaridad
Ayuda: ver de utilizar una matriz de similaridad para visualizar todas las similaridades par-a-par juntas.
¿Para qué elección de d resulta más conveniente hacer el análisis? ¿Cómo se conecta esto con los valores singulares de X? ¿Qué conclusiones puede sacar al respecto?

3. Los datos X vienen acompañados de una variable dependiente respuesta o etiquetas llamada Y (archivo y.txt) estructurada como un vector n x 1 para cada muestra. Queremos encontrar el vector ® y modelar
linealmente el problema que minimice la norma
|XB -yll2
de manera tal de poder predecir con XB - ý lo mejor posible a las etiquetas y, es decir, minimizar el error de predicción utilizando todas las variables iniciales. Resolviendo el problema de cuadrados mínimos en el espacio original X, que peso se le asigna a cada dimensión original si observamos el
vector B?

4. Usando la representacion aprendida con PCA y d - 2: mejora la predicción || ZB - y lle en comparacion a no realizar reduccion de dimensionalidad? Cuales muestras son las de mejor predicción con el mejor modelo?
"""

X_ = pd.read_csv('/Users/belengotz/Desktop/dataset_x_y/dataset01.csv')
X_ = X_.drop(X_.columns[0], axis = 1)
X = StandardScaler().fit_transform(X_)



U, S, Vt = np.linalg.svd(X, full_matrices=False)

#VISUALIZAR A
plt.subplot(1, 1, 1)
plt.imshow(X, interpolation='nearest', aspect='auto', cmap='viridis')
plt.title('Matriz original (X)')
plt.colorbar()

#SCATTERPLOT 2D
V_2 = Vt[:2, :]  
Z2 = X @ V_2.T
plt.figure(figsize=(12, 8))  
plt.scatter(Z2[:, 0], Z2[:, 1], s=50, c='#6495ED', marker='o')
plt.title(f'Proyección en 2 dimensiones (d={2})')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.show()


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

#PARÁMETROS A EVALUAR
d_values = [2, 6, 10, X.shape[1]] 
#sigma = [1, 5, 10, 50, 100, 1000] #IR CAMBIANDO
sigma = [1];
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
    plt.show()

Y_ = np.loadtxt('/Users/belengotz/Desktop/dataset_x_y/y1.txt')
Y_centered = Y_ - np.mean(Y_)

""" OLS """
#X_pseudo_inverse = np.linalg.inv(X.T @ X) @ X.T #(X^T X)^{-1} X^T
X_pseudo_inverse = np.linalg.pinv(X)

B = X_pseudo_inverse @ Y_centered #calculo B usando la pseudo-inversa

""" ECM """
mse = np.mean((X @ B - Y_centered) ** 2)
print(f"Error cuadrático medio del modelo: {mse}")

# Visualización de los pesos asignados a cada dimensión original
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(B) + 1), B)
plt.xlabel("Dimensión original")
plt.ylabel("Peso asignado (B)")
plt.title("Pesos asignados a cada dimensión original en el modelo de cuadrados mínimos")
plt.show()


explained_variances = np.cumsum(S ** 2) / np.sum(S ** 2)  # Variancia acumulada explicada
threshold = 0.90  # 90% para un error promedio del 10%
min_components = np.argmax(explained_variances >= threshold) + 1

print(f"Número de componentes necesarios para asegurar un error promedio del 10%: {min_components}")