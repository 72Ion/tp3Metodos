from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt

A = imread('/Users/belengotz/Desktop/TP 03 dataset imagenes/IMG_0568.jpg')
print(A.shape)
plt.imshow(A)
print(np.max(A))
print(np.min(A))


if len(A.shape) == 3:
    print(A[:2, :2, 0])
else:
    print(A[:2, :2])

#Convierto a escala de grises promediando los 3 canales RGB
if len(A.shape) == 3:
    X = np.mean(A, axis=-1)
else:
    X = A

plt.imshow(X, cmap = 'gray')

# Cálculo de SVD
U, S, VT = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)

print('Dim(U)=', U.shape, '\nDim(S)=', S.shape, '\nDim(V^t)=', VT.shape)
print('Rango de X = 360')

# Rangos de truncación
ks = [1, 3, 5, 20, 50, 100]

# Factor de Compresión
def factor_compresion(k, X):
    m, n = X.shape
    peso_pixel = 8  # en bytes (doble precisión)
    peso_imagen = m * n * peso_pixel
    peso_svd_trunc = (m + n + 1) * k * peso_pixel
    return peso_svd_trunc / peso_imagen

# Error de reconstrucción (Norma de Frobenius)
def error_reconstruccion(X, Xapprox):
    return np.linalg.norm(X - Xapprox, 'fro') / np.linalg.norm(X, 'fro')

# Visualización de imágenes comprimidas y cálculo de errores
fig, axs = plt.subplots(len(ks) // 3, len(ks) // 2, figsize=(14, 8))
axs = axs.flatten()
errores = []

for ax, d in zip(axs, ks):
    Xapprox = U[:, :d] @ S[:d, :d] @ VT[:d, :]
    error = error_reconstruccion(X, Xapprox)
    errores.append(error)
    factor = factor_compresion(d, X)
    ax.imshow(Xapprox, cmap='gray')
    ax.set_title(f'd = {d}, Compresión = {round(factor * 100, 2)}%, Error = {round(error * 100, 2)}%')
    if d == 50:
        plt.imsave('imagen_procesada_d50.png', Xapprox, cmap='gray')

plt.show()

# Gráfico de error en función de k
plt.figure(figsize=(10, 5))
plt.plot(ks, errores, '-o')
plt.axhline(0.1, color='r', linestyle='--', label='Error del 10%')
plt.xlabel('d')
plt.ylabel('Error de reconstrucción (%)')
plt.yscale('log')
plt.title('Error de Reconstrucción vs. Número de Componentes d')
plt.legend()
plt.show()

# Gráficos de valores singulares y energía acumulada
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(np.diag(S), '-o')
ax1.set_yscale('log')
ax1.set_title('Valores Singulares')
ax1.set_xlabel(r'$d$')
ax1.set_ylabel(r'$\sigma$')
ax2.plot(np.cumsum(np.diag(S)) / np.sum(np.diag(S)))
ax2.set_xlabel(r'$d$')
ax2.set_ylabel(r'$\mathrm{cumsum}(\sigma)/\mathrm{sum}(\sigma)$')
ax2.set_title('Energía Acumulada')

plt.show()

#hola