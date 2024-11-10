import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import os

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

img_directory = '/Users/belengotz/Desktop/tp3Metodos/TP 03 dataset imagenes'
img_size = 28 
images_matrix = load_images(img_directory, img_size)

def svd_compression(images, d):
    #SVD
    U, S, Vt = np.linalg.svd(images, full_matrices=False)
    
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    
    compressed_images = U_d @ S_d @ Vt_d
    return compressed_images

def visualize_images(original_images, compressed_images, img_size, title=None):
    num_images = len(original_images)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i in range(num_images):
        ax = axes[0, i]
        ax.imshow(original_images[i].reshape(img_size, img_size), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title('Original')
        
        ax = axes[1, i]
        ax.imshow(compressed_images[i].reshape(img_size, img_size), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title('Compressed')
    
    if title:
        plt.suptitle(title)
    
    plt.show()

def calculate_error(original, compressed):
    error = np.linalg.norm(original - compressed, 'fro') / np.linalg.norm(original, 'fro')
    return error * 100 

d_values = [1, 5, 10, 15, 20] 
errors = []

for d in d_values:
    compressed_images = svd_compression(images_matrix, d)
    error = calculate_error(images_matrix, compressed_images)
    errors.append(error)
    
    #visualize_images(images_matrix, compressed_images, img_size, title=f"Valor de d: {d}, Error de compresión: {error:.2f}%")

target_error = 10 
for d in range(1, images_matrix.shape[1] + 1):
    compressed_images = svd_compression(images_matrix, d)
    error = calculate_error(images_matrix, compressed_images)
    if error <= target_error:
        print(f"Valor mínimo de d para asegurar un error ≤ {target_error}%: {d}")
        break

errors_range = []
d_range = range(1, 20)

for d in d_range:
    compressed_images = svd_compression(images_matrix, d)
    error_range = calculate_error(images_matrix, compressed_images)
    errors_range.append(error_range)
    

plt.figure(figsize=(8, 5))
plt.plot(d_range, errors_range, marker='o', color='b')
#plt.yscale('log')
plt.xlabel('Dimensión d')
plt.ylabel('Error de compresión (%)')
plt.title('Evolución del error de compresión con respecto a d')
plt.grid()
#plt.show()


"""

agregar interprestacion de por que se apilan las imagenes en el punto 1

la idea: 
uno podria agarrar imagen x imagen y hacer svd sobre eso, pero

graficar los autovectores

aprender una base de todas las imagenes juntas?????

"""

# Mostrar autovalores de la matriz SVD
U, S, Vt = np.linalg.svd(images_matrix, full_matrices=False)
autovalores = S

# Graficar los autovalores
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(autovalores) + 1), autovalores, marker='o', color='g')
plt.xlabel('Componente Principal')
plt.ylabel('Autovalor')
plt.title('Autovalores de la matriz de imágenes')
plt.grid()
plt.show()



top_d = 5  # Número de autovectores a usar (por ejemplo, los 5 más importantes)
top_Vt = Vt[top_d:, :]

# Multiplicar los autovectores seleccionados por la matriz de imágenes
reconstructed_images = images_matrix @ top_Vt.T  # Proyección de las imágenes en los autovectores

# Visualizar los primeros 5 autovectores más importantes
num_vects = 5
for i in range(num_vects):
    plt.subplot(1, num_vects, i+1)
    plt.imshow(Vt[i].reshape(img_size, img_size), cmap='gray')
    plt.axis('off')
    plt.title(f'Autovector {i+1}')
plt.show()

for i in range(num_vects):
    plt.subplot(1, num_vects, i+1)
    # Seleccionamos las últimas filas de Vt, es decir, los últimos autovectores
    plt.imshow(Vt[-(i+1)].reshape(img_size, img_size), cmap='gray')
    plt.axis('off')
    plt.title(f'Autovector {-(i+1)}')
plt.show()


