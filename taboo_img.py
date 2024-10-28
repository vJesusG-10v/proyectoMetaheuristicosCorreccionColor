import numpy as np
import cv2
from PIL import Image

#Metrica SNR para la optimizacion
def SNR(original, processed):
    
    noise_initial = processed - original
    signal = np.mean(original ** 2)
    noise = np.mean(noise_initial ** 2)
    
    if noise == 0:
        return float('inf')  
    return 20 * np.log10(signal / noise)

#Funion para aplicar las mascaras
def apply_filters(image, blur_mask, sharp_mask):
    
    channels = cv2.split(image)
    processed_channels = []
    
    for channel in channels:
        
        sharp = cv2.filter2D(channel, 0, sharp_mask)
        blur = cv2.filter2D(sharp, 0, blur_mask)
        processed_channels.append(blur)

    return cv2.merge(processed_channels)

#Funcion para generar vecinos para la mascara de afilado 
def generate_neighbors_sharp(mask, step=0.4): 
    neighbors = []
    for i in range(len(mask)):
        for delta in [-step, step]:
            new_mask = mask.copy()
            new_mask[i] = np.clip(new_mask[i] + delta, -1, 1.5)
            neighbors.append(new_mask)
    return neighbors

#Funcion para generar vecinos para la mascara de suavizado
def generate_neighbors_blur(mask, step=0.8): 
    neighbors = []
    center_index = len(mask) // 2

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            for delta in [-step, step]:
                new_mask = mask.copy()

                #Preservar un centro "Mayor"
                if i == center_index and j == center_index:
                    new_mask[i, j] = np.clip(new_mask[i, j] + delta, 0.7, 1.0)
                else:
                    # Asegurar que los vecinos del centro varíen dentro de un rango menor
                    new_mask[i, j] = np.clip(new_mask[i, j] + delta, 0.0, 0.5)
                
                #Normalizar la mascara para que la suma sea 1
                total_sum = np.sum(new_mask)
                if total_sum > 0:
                    new_mask /= total_sum
                else:
                    #Si la suma es 0 asignar una mascara uniforme
                    new_mask = np.full_like(new_mask, 1 / len(new_mask))
                
                neighbors.append(new_mask)
    
    return neighbors

#Metaheuristico taboo para optimizar valores de las mascaras en base a la metrica SNR
def taboo_search(initial_blur_mask, initial_sharp_mask, iterations, taboo_size, image, update_progress):
    # Inicializa variables del algoritmo
    actual_blur_mask = initial_blur_mask
    actual_sharp_mask = initial_sharp_mask
    best_blur_mask = actual_blur_mask
    best_sharp_mask = actual_sharp_mask
    best_snr = float('-inf')
    tabu_list = []

    # Bucle principal del algoritmo
    for iteration in range(iterations):
        # Generación de vecinos para ambas máscaras
        blur_neighbors = generate_neighbors_blur(actual_blur_mask)
        sharp_neighbors = generate_neighbors_sharp(actual_sharp_mask)

        best_neighbor_snr = float('-inf')
        best_neighbor = (None, None)

        # Búsqueda de taboos
        for b_neighbor in blur_neighbors:
            for s_neighbor in sharp_neighbors:
                if (b_neighbor.tolist(), s_neighbor.tolist()) in tabu_list:
                    continue  

                # Aplicación de las máscaras generadas
                processed_image = apply_filters(image, b_neighbor, s_neighbor)
                # Cálculo del SNR
                snr = SNR(image, processed_image)

                if snr > best_neighbor_snr:
                    best_neighbor = (b_neighbor, s_neighbor)
                    best_neighbor_snr = snr

        # Evaluar resultados entre la iteración anterior y la actual
        if best_neighbor[0] is not None and best_neighbor[1] is not None:
            actual_blur_mask, actual_sharp_mask = best_neighbor
            if best_neighbor_snr > best_snr:
                best_blur_mask = actual_blur_mask
                best_sharp_mask = actual_sharp_mask
                best_snr = best_neighbor_snr

        # Agregar a la lista tabu
        tabu_list.append((actual_blur_mask.tolist(), actual_sharp_mask.tolist()))
        if len(tabu_list) > taboo_size:
            tabu_list.pop(0)

        # Actualizar el progreso después de cada iteración
        update_progress(iteration + 1, iterations, "Afilando su imagen")  # Actualizar progreso

    return best_blur_mask, best_sharp_mask, best_snr

#Funcion para inicializar aleatoriamente ambas mascaras
def init_masks(mask_size):
    #Inicializar la máscara de suavizado
    blur_mask = np.random.rand(mask_size, mask_size)
    center_index = mask_size // 2
    #Aumentar el valor del centro 
    blur_mask[center_index, center_index] = np.random.uniform(0.7, 1.0) * 2
    
    for i in range(mask_size):
        for j in range(mask_size):
            if (i, j) != (center_index, center_index):
                blur_mask[i, j] = np.random.uniform(0.0, 1.0)
    
    #Normalizar la mascara para que la suma sea 1
    blur_mask /= np.sum(blur_mask)
    print("Mascara de suavizado inicializada:\n", blur_mask)

    #Inicializar la mascara de afilado
    sharp_mask = np.zeros((mask_size, mask_size))
    sharp_mask[center_index, center_index] = np.random.uniform(1, 1.5) 
    for i in range(mask_size):
        for j in range(mask_size):
            if (i, j) != (center_index, center_index):
                sharp_mask[i, j] = np.random.uniform(-1.0, 1.5)
    print("Mascara de afilado inicializada:\n", sharp_mask)
    
    return blur_mask, sharp_mask

def sharpen_image(image_pil, update_progress, sharpen_iters):
    # Convertir de PIL a NumPy (OpenCV)
    image = np.array(image_pil)

    # Configuración de las máscaras y parámetros
    blur_mask, sharp_mask = init_masks(5)
    max_iter = sharpen_iters  # Usar el valor proporcionado por el usuario desde la interfaz
    taboo_size = 10

    # Ejecución del algoritmo de suavizado
    best_blur_mask, best_sharp_mask, best_snr = taboo_search(blur_mask, sharp_mask, max_iter, taboo_size, image, update_progress)

    # Aplicación de los filtros a la imagen
    final_image = apply_filters(image, best_blur_mask, best_sharp_mask)

    #Se cambia el formato de BGR a RGB para poder mostrarse correctamente
    final_image_cv = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    # Convertir la imagen final a PIL antes de devolver
    final_image_pil = Image.fromarray(final_image_cv)

    return final_image_pil, best_blur_mask, best_sharp_mask, best_snr