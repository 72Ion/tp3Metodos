import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Configuración para estilos visuales
sns.set(style="whitegrid")

# Asumiendo que tienes las siguientes variables predefinidas:
# X: Matriz de características original
# Y: Vector de salida
# pca: Modelo PCA ajustado sobre X
# explained_variance_ratio: Ratio de varianza explicada por componente (pca.explained_variance_ratio_)
# beta: Coeficientes del modelo de regresión
# prediction_errors: Array con los errores de predicción por muestra
# similarity_matrix: Matriz de similitud en el espacio reducido

# 1. Gráfico de Varianza Explicada Acumulada
def plot_cumulative_variance(explained_variance_ratio):
    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='b')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Varianza Explicada')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Varianza Explicada Acumulada en Función del Número de Componentes')
    plt.legend()
    plt.show()

# Ejecutar el gráfico de varianza explicada acumulada
plot_cumulative_variance(pca.explained_variance_ratio_)

# 2. Distribución de los Pesos (Beta) en el Modelo de Cuadrados Mínimos
def plot_beta_distribution(beta, num_highlight=10):
    plt.figure(figsize=(10, 6))
    beta_abs_sorted_indices = np.argsort(np.abs(beta))[::-1]
    top_beta_indices = beta_abs_sorted_indices[:num_highlight]
    colors = ['red' if i in top_beta_indices else 'blue' for i in range(len(beta))]
    
    plt.bar(range(len(beta)), beta, color=colors)
    plt.xlabel('Dimensiones')
    plt.ylabel('Pesos (Beta)')
    plt.title('Distribución de los Pesos (Beta) en el Modelo de Cuadrados Mínimos')
    plt.show()

# Asumimos que beta son los coeficientes del modelo lineal
model = LinearRegression().fit(X, Y)
beta = model.coef_

# Ejecutar el gráfico de distribución de los pesos
plot_beta_distribution(beta)

# 3. Gráfico de Errores de Predicción en el Espacio Reducido
def plot_prediction_errors(prediction_errors, top_n=5):
    sorted_errors_indices = np.argsort(prediction_errors)
    top_errors = prediction_errors[sorted_errors_indices[:top_n]]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(prediction_errors)), prediction_errors, 'o', color='grey', alpha=0.5, label='Errores de Predicción')
    plt.plot(sorted_errors_indices[:top_n], top_errors, 'o', color='red', label='Top 5 Menores Errores')
    plt.xlabel('Índice de Muestra')
    plt.ylabel('Error de Predicción')
    plt.title(f'Top {top_n} Errores de Predicción Mínimos')
    plt.legend()
    plt.show()

# Supón que prediction_errors es un array con los errores de predicción para cada muestra
# Ejecutar el gráfico de errores de predicción
plot_prediction_errors(prediction_errors)

# 4. Distribución de los Valores en la Matriz de Similitud
def plot_similarity_distribution(similarity_matrix, d_values):
    fig, axes = plt.subplots(1, len(d_values), figsize=(16, 6), sharey=True)
    
    for i, d in enumerate(d_values):
        sns.histplot(similarity_matrix[d].flatten(), bins=20, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribución de Similitud para d={d}')
        axes[i].set_xlabel('Valor de Similitud')
    
    axes[0].set_ylabel('Frecuencia')
    plt.suptitle('Distribución de los Valores en la Matriz de Similitud para Diferentes d')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Supón que similarity_matrix es un diccionario con matrices de similitud para distintos valores de d
d_values = [2, 6, 10, 207]  # Valores de d a analizar
similarity_matrix = {d: np.random.rand(100, 100) for d in d_values}  # Ejemplo de matriz aleatoria para pruebas

# Ejecutar el gráfico de distribución de valores en la matriz de similitud
plot_similarity_distribution(similarity_matrix, d_values)
