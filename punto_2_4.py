import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Cargar los datos X e Y
X = pd.read_csv('/Users/belengotz/Desktop/dataset_x_y/dataset01.csv').values
Y = pd.read_csv('/Users/belengotz/Desktop/dataset_x_y/y1.txt', header=None).values.flatten()

# Paso 1: Estandarizar X y centrar Y
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Y_centered = Y - np.mean(Y)  # Centrar Y (restar la media)

# Paso 2: Realizar PCA para reducir la dimensionalidad a d=2
pca = PCA(n_components=2)
Z = pca.fit_transform(X_scaled)  # Proyección en el espacio reducido

# Paso 3: Resolver el problema de cuadrados mínimos para la predicción en el espacio original (X)
# Pseudo inversa en el espacio original
B_original = np.linalg.pinv(X_scaled).dot(Y_centered)

# Paso 4: Realizar predicción en el espacio original y calcular el error cuadrático medio (ECM)
Y_pred_original = X_scaled.dot(B_original)
mse_original = mean_squared_error(Y_centered, Y_pred_original)
print(f"Error cuadrático medio (ECM) en el espacio original: {mse_original}")

# Paso 5: Resolver el problema de cuadrados mínimos para la predicción en el espacio reducido (Z)
# Pseudo inversa en el espacio reducido
B_reduced = np.linalg.pinv(Z).dot(Y_centered)

# Paso 6: Realizar predicción en el espacio reducido y calcular el error cuadrático medio (ECM)
Y_pred_reduced = Z.dot(B_reduced)
mse_reduced = mean_squared_error(Y_centered, Y_pred_reduced)
print(f"Error cuadrático medio (ECM) en el espacio reducido (d=2): {mse_reduced}")

# Paso 7: Comparar el ECM y determinar qué modelo mejora la predicción
if mse_reduced < mse_original:
    print("La reducción de dimensionalidad mejora la predicción.")
else:
    print("La reducción de dimensionalidad no mejora la predicción.")

# Paso 8: Identificar las muestras con mejor predicción (menor error) en el modelo reducido
errors_reduced = np.abs(Y_centered - Y_pred_reduced)
best_predictions_idx = np.argsort(errors_reduced)[:5]  # Las 5 muestras con el menor error
print("Índices de las muestras con la mejor predicción en el espacio reducido:")
print(best_predictions_idx)

# Paso 9: Graficar las muestras con el mejor modelo de predicción
plt.figure(figsize=(10, 6))
plt.scatter(Z[:, 0], Z[:, 1], c=errors_reduced, cmap='viridis', s=50)
plt.colorbar(label="Error de predicción")
plt.title("Distribución de las muestras según el error de predicción en el espacio reducido (d=2)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.scatter(Z[best_predictions_idx, 0], Z[best_predictions_idx, 1], color='red', label='Mejor predicción', zorder=5)
plt.legend()
plt.show()


