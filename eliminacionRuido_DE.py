import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk

# Definir la función del filtro Gaussiano manualmente
def apply_gaussian_filter(image, kernel_size, sigma):
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel)
    return cv2.filter2D(image, -1, kernel)

def objective_function(coeffs, noisy_image):
    kernel_size = int(coeffs[0])
    sigma = coeffs[1]
    
    if kernel_size % 2 == 0:  # Asegurarse de que el tamaño del kernel sea impar
        kernel_size += 1
    
    filtered_image = apply_gaussian_filter(noisy_image, kernel_size, sigma)
    
    # Usamos un suavizado básico como referencia
    baseline_filtered = apply_gaussian_filter(noisy_image, 7, 2.0)  # Suavizado básico con kernel de 7 y sigma 2
    
    # Minimizar la diferencia entre la imagen suavizada y la imagen filtrada
    return np.mean((baseline_filtered - filtered_image) ** 2)

# Algoritmo de evolución diferencial (DE)
def differential_evolution(func, bounds, pop_size, max_iter, F, CR, noisy_image, update_progress):
    degree = bounds.shape[0]
    
    # Inicialización aleatoria de la población
    pop = np.random.rand(pop_size, degree)
    pop = bounds[:, 0] + pop * (bounds[:, 1] - bounds[:, 0])  # Escalar según los límites
    fitness = np.array([func(ind, noisy_image) for ind in pop])

    # Iteraciones del algoritmo DE
    for gen in range(max_iter):
        # Actualizar progreso incluso en la primera iteración
        update_progress(gen + 1, max_iter, "Suavizando imagen")  # Actualizar progreso

        # Realizar las operaciones del algoritmo DE
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            
            mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
            cross_points = np.random.rand(degree) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, degree)] = True
            trial = np.where(cross_points, mutant, pop[i])
            
            trial_fitness = func(trial, noisy_image)
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                pop[i] = trial

    best_idx = np.argmin(fitness)
    best_params = pop[best_idx]
    return best_params


# Función para aplicar evolución diferencial a una imagen proporcionada
def process_image_with_de(noisy_image, update_progress, blur_iters):
    # Parámetros de DE
    pop_size = 50
    max_iter = blur_iters  # Usar el valor proporcionado por el usuario desde la interfaz
    F = 0.7
    CR = 0.8
    bounds = np.array([[5, 51],    # Tamaño del kernel: entre 5 y 51 (impar)
                       [0.1, 15]]) # Sigma: entre 0.1 y 15

    # Aplicar evolución diferencial para encontrar el mejor filtro
    best_params = differential_evolution(objective_function, bounds, pop_size, max_iter, F, CR, noisy_image, update_progress)

    # Usar los mejores parámetros para filtrar la imagen ruidosa
    best_kernel_size = int(best_params[0])
    if best_kernel_size % 2 == 0:
        best_kernel_size += 1
    best_sigma = best_params[1]
    filtered_image = apply_gaussian_filter(noisy_image, best_kernel_size, best_sigma)

    return noisy_image, filtered_image  # Devolver ambas imágenes para mostrarlas en el frontend



image_references = {}
def show_result_window(noisy_image, filtered_image):
    # Crear ventana de Tkinter
    window = tk.Tk()
    window.title("Resultados de filtrado")
    window.resizable(False, False)

    # Convertir las imágenes de OpenCV a PIL para poder mostrarlas en Tkinter
    noisy_image_pil = Image.fromarray(noisy_image)
    filtered_image_pil = Image.fromarray(filtered_image)

    # Convertir PIL a ImageTk
    image_references['noisy'] = ImageTk.PhotoImage(noisy_image_pil)  # Guardar referencia
    image_references['filtered'] = ImageTk.PhotoImage(filtered_image_pil)  # Guardar referencia

    # Crear etiquetas para las imágenes
    label_noisy = tk.Label(window, image=image_references['noisy'])
    label_noisy.grid(row=0, column=0)
    label_filtered = tk.Label(window, image=image_references['filtered'])
    label_filtered.grid(row=0, column=1)

    # Mantener referencias a las imágenes para evitar que sean recolectadas
    label_noisy.image = image_references['noisy']  # Guardar referencia
    label_filtered.image = image_references['filtered']  # Guardar referencia

    # Ejecutar el loop de Tkinter
    window.mainloop()