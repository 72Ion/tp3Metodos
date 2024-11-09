import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import os

# Cargar y preprocesar las imágenes
def load_images(directory, img_size):
    images = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            img = imread(filepath, as_gray=True)
            img_resized = resize(img, (img_size, img_size), anti_aliasing=True)
            images.append(img_resized.flatten())
        except (IOError, ValueError):
            print(f"Archivo no válido o no se pudo leer: {filename}")
            continue
    return np.array(images)

# Ruta a la carpeta que contiene las imágenes y tamaño deseado de cada imagen (p x p)
img_directory = '/Users/belengotz/Desktop/tp3Metodos/TP 03 dataset imagenes'
img_size = 28  # Tamaño de las imágenes (28x28 en este caso)
images_matrix = load_images(img_directory, img_size)

# SVD para reducir la dimensionalidad
def svd_compression(images, d):
    # Aplicar SVD
    U, S, Vt = np.linalg.svd(images, full_matrices=False)
    
    # Mantener solo los primeros d valores singulares y vectores
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    
    # Reconstruir la matriz comprimida
    compressed_images = U_d @ S_d @ Vt_d
    return compressed_images

# Visualizar imágenes originales y comprimidas
def visualize_images(original, compressed, img_size, num_images=5):
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        # Imagen original
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original[i].reshape(img_size, img_size), cmap='gray')
        plt.axis('off')
        plt.title("Original")

        # Imagen comprimida
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(compressed[i].reshape(img_size, img_size), cmap='gray')
        plt.axis('off')
        plt.title("Comprimida")
    plt.show()

# Calcular el error de compresión
def calculate_error(original, compressed):
    error = np.linalg.norm(original - compressed, 'fro') / np.linalg.norm(original, 'fro')
    return error * 100  # Error como porcentaje

# Lista de valores de d para probar
d_values = [5, 10, 15, 20]  # Ajustar según tus necesidades
errors = []

# Probar distintos valores de d y visualizar resultados
for d in d_values:
    compressed_images = svd_compression(images_matrix, d)
    error = calculate_error(images_matrix, compressed_images)
    errors.append(error)
    
    print(f"Valor de d: {d}, Error de compresión: {error:.2f}%")
    visualize_images(images_matrix, compressed_images, img_size)

# Encontrar el valor mínimo de d que asegura un error ≤ 10%
target_error = 10  # Porcentaje de error deseado
for d in range(1, images_matrix.shape[1] + 1):
    compressed_images = svd_compression(images_matrix, d)
    error = calculate_error(images_matrix, compressed_images)
    if error <= target_error:
        print(f"Valor mínimo de d para asegurar un error ≤ {target_error}%: {d}")
        break

# Graficar la evolución del error con respecto a d
plt.figure(figsize=(8, 5))
plt.plot(d_values, errors, marker='o', color='b')
plt.xlabel('Dimensión d')
plt.ylabel('Error de compresión (%)')
plt.title('Evolución del error de compresión con respecto a d')
plt.grid()
plt.show()
