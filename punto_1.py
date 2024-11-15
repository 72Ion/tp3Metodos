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


# Verificar las dimensiones de las imágenes redimensionadas
print(f"Dimensiones de las imágenes redimensionadas: {images_matrix.shape}")

def calculate_error(original, compressed):
    error = np.linalg.norm(original - compressed, 'fro') / np.linalg.norm(original, 'fro')
    return error * 100 


def visualize_images_matrix(images_matrix, img_size):
    num_images = images_matrix.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images_matrix[i].reshape(img_size, img_size), cmap='gray')
        ax.axis('off')
    
    # Hide any remaining axes if the number of images is not a perfect square
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
visualize_images_matrix(images_matrix, img_size)
print(f"Las dimensiones de cada imagen son: {img_size}x{img_size} píxeles")



d_values = [1, 5, 10, 15, 20] 
errors = []

for d in d_values:
    compressed_images = svd_compression(images_matrix, d)
    error = calculate_error(images_matrix, compressed_images)
    errors.append(error)
    
    visualize_images(images_matrix, compressed_images, img_size, title=f"Valor de d: {d}, Error de compresión: {error:.2f}%")

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
plt.show()


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



def calculate_average_error(original_images, compressed_images):
    """
    Calcula el error promedio de compresión entre las imágenes originales y comprimidas.
    """
    num_images = original_images.shape[0]
    total_error = 0
    
    for i in range(num_images):
        error = np.linalg.norm(original_images[i] - compressed_images[i]) / np.linalg.norm(original_images[i])
        total_error += error
    
    average_error = (total_error / num_images) * 100  # Convertir a porcentaje
    return average_error

def find_optimal_d_average(images_matrix, target_error=10):
    """
    Encuentra el valor óptimo de d que asegura un error promedio de compresión ≤ target_error.
    """
    for d in range(1, images_matrix.shape[1] + 1):
        compressed_images = svd_compression(images_matrix, d)
        average_error = calculate_average_error(images_matrix, compressed_images)
        if average_error <= target_error:
            return d, average_error
    return images_matrix.shape[1], average_error  # En caso de que no se alcance el error deseado

# Calcular el valor óptimo de d para un error promedio ≤ 10%
optimal_d_avg, optimal_avg_error = find_optimal_d_average(images_matrix, target_error=10)
print(f"Valor mínimo de d para asegurar un error promedio ≤ 10%: {optimal_d_avg}, con un error promedio de compresión de {optimal_avg_error:.2f}%")

# Visualizar la evolución del error promedio en función de d
average_errors_range = []
d_range = range(1, 16)

for d in d_range:
    compressed_images = svd_compression(images_matrix, d)
    avg_error = calculate_average_error(images_matrix, compressed_images)
    average_errors_range.append(avg_error)

plt.figure(figsize=(8, 5))
plt.plot(d_range, average_errors_range, marker='o', color='b')
plt.axvline(optimal_d_avg, color='r', linestyle='--', label=f'Valor óptimo de d: {optimal_d_avg}')
plt.xlabel('Dimensión d')
plt.ylabel('Error de compresión promedio (%)')
plt.title('Evolución del error promedio de compresión con respecto a d')
plt.legend()
plt.grid()
plt.show()
