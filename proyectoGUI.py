import customtkinter as ctk
import cv2
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.messagebox as messagebox
import re

from eliminacionRuido_DE import process_image_with_de
from taboo_img import sharpen_image  

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.noisy_image_tk = None
        self.filtered_image_tk = None
        self.sharpened_image_tk = None  # Para la imagen afilada
        self.progress_window = None

        # Variables para almacenar del usuario
        self.blur_iterations = ctk.StringVar()
        self.sharp_iterations = ctk.StringVar()
        self.image_path = None  # Para verificar si el usuario subió una imagen

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            messagebox.showinfo("Imagen Cargada", "La imagen ha sido cargada exitosamente.")

    def process_image(self):
        original_image = cv2.imread(self.image_path)
        self.root.withdraw()  # Ocultar la ventana principal
        self.show_progress_window()

        # Procesar la imagen con el filtro Gaussiano y el algoritmo de afilado
        noisy_image, filtered_image = process_image_with_de(original_image, self.update_progress, int(self.blur_iterations.get()))
        sharpened_image, best_blur_mask, best_sharp_mask, best_snr = sharpen_image(noisy_image, self.update_progress, int(self.sharp_iterations.get()))

        self.progress_window.destroy()  # Cerrar la ventana de progreso
        self.show_result_window(noisy_image, filtered_image, sharpened_image, best_blur_mask, best_sharp_mask, best_snr)  # Mostrar la ventana con las imágenes

    def show_progress_window(self):
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Procesando...")

        self.processing_message = tk.Label(self.progress_window, text="Su imagen está siendo procesada")
        self.processing_message.pack(pady=10)

        # Barra de progreso
        self.progress_bar = ttk.Progressbar(self.progress_window, orient='horizontal', length=300, mode='determinate')
        self.progress_bar.pack(pady=20)

        # Etiqueta de progreso
        self.progress_label = tk.Label(self.progress_window, text='Progreso: (0.00%)')
        self.progress_label.pack(pady=10)

    def validate_entries(self):
       
        if not self.blur_iterations.get().strip() or not self.sharp_iterations.get().strip():
            messagebox.showerror("Error", "Los campos de iteraciones no pueden estar vacíos.")
            return False

        # Validar que los valores sean números positivos usando regex
        blur_iters = self.blur_iterations.get().strip()
        sharp_iters = self.sharp_iterations.get().strip()

        if not re.match(r'^\d+$', blur_iters) or not re.match(r'^\d+$', sharp_iters):
            messagebox.showerror("Error", "Los campos de iteraciones deben contener solo números positivos.")
            return False

        blur_iters = int(blur_iters)
        sharp_iters = int(sharp_iters)

        if blur_iters <= 0 or sharp_iters <= 0:
            messagebox.showerror("Error", "Las iteraciones deben ser números positivos.")
            return False

        # Verificar si el usuario ha subido una imagen
        if not self.image_path:
            messagebox.showerror("Error", "Debes subir una imagen.")
            return False

        # Advertir si el valor supera 100
        if blur_iters > 100 or sharp_iters > 100:
            return messagebox.askyesno("Advertencia", "Has ingresado un valor mayor a 100, esto puede tardar mucho tiempo. ¿Estás seguro de que quieres continuar?")

        return True

    def on_continue(self):
        if self.validate_entries():
            self.process_image() # Si la validación es correcta, proceder a cargar la imagen

    def update_progress(self, current, total, message):
        self.progress_bar['value'] = (current / total) * 100  # Actualiza el valor de la barra
        self.progress_label.config(text=f'Progreso: {current}/{total} - {message}')  # Actualiza el texto de progreso
        self.progress_window.update_idletasks()  # Fuerza a la ventana a actualizarse
    

    def show_result_window(self, noisy_image, filtered_image, sharpened_image, best_blur_mask, best_sharp_mask, best_snr):
        window = tk.Toplevel(self.root)  # Usar Toplevel para crear una nueva ventana
        window.title("Resultados de filtrado")

        #Se cambia el formato de BGR a RGB para poder mostrarse correctamente
        noisy_image_cv = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        filtered_image_cv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

        # Convertir las imágenes de OpenCV a PIL para poder mostrarlas en Tkinter
        noisy_image_pil = Image.fromarray(noisy_image_cv)
        filtered_image_pil = Image.fromarray(filtered_image_cv)

        # Verificar si sharpened_image es un objeto Image o un array
        if isinstance(sharpened_image, Image.Image):
            sharpened_image_pil = sharpened_image  # Ya es una imagen PIL
        else:
            # Si es un array, conviértelo a PIL
            sharpened_image_cv = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)
            sharpened_image_pil = Image.fromarray(sharpened_image_cv)

        #self.noisy_image_tk = cv2.cvtColor(self.noisy_image_tk, cv2.COLOR_BG2RGB)

        # Convertir PIL a ImageTk
        self.noisy_image_tk = ImageTk.PhotoImage(noisy_image_pil)
        self.filtered_image_tk = ImageTk.PhotoImage(filtered_image_pil)
        self.sharpened_image_tk = ImageTk.PhotoImage(sharpened_image_pil)

        # Crear títulos para las imágenes
        label_title_noisy = tk.Label(window, text="Imagen original", font=("Helvetica", 16))
        label_title_noisy.grid(row=0, column=0, pady=(10, 0))

        label_title_filtered = tk.Label(window, text="Imagen suavizada", font=("Helvetica", 16))
        label_title_filtered.grid(row=0, column=1, pady=(10, 0))

        label_title_sharpened = tk.Label(window, text="Imagen afilada", font=("Helvetica", 16))
        label_title_sharpened.grid(row=0, column=2, pady=(10, 0))

        # Crear etiquetas para las imágenes
        label_noisy = tk.Label(window, image=self.noisy_image_tk)
        label_noisy.grid(row=1, column=0, padx=10, pady=10)

        label_filtered = tk.Label(window, image=self.filtered_image_tk)
        label_filtered.grid(row=1, column=1, padx=10, pady=10)

        label_sharpened = tk.Label(window, image=self.sharpened_image_tk)
        label_sharpened.grid(row=1, column=2, padx=10, pady=10)

        # Mantener referencias a las imágenes para evitar que sean recolectadas
        label_noisy.image = self.noisy_image_tk
        label_filtered.image = self.filtered_image_tk
        label_sharpened.image = self.sharpened_image_tk

        # Imprimir información de las mejores máscaras y SNR en la terminal
        print("Mejor máscara de suavizado:\n", best_blur_mask)
        print("Mejor máscara de afilado:\n", best_sharp_mask)
        print("SNR de la mejor solución:", best_snr)


# Crear la ventana principal
app = ctk.CTk()
app.title("Suavizado de imágenes")
app.geometry("600x400")
app.resizable(False, False)

# Crear la instancia de ImageApp
image_app = ImageApp(app)

# Descripción breve del proyecto
label_description = ctk.CTkLabel(app, text="\nEste programa aplica un suavizado optimizado a la imagen utilizando Evolución Diferencial.\nSeleccione una imagen para comenzar.\n\nPara resultados más rápidos se sugiere usar una imagen a blanco y negro.", anchor="center")
label_description.pack(pady=20, padx=20)

# Botón para cargar imagen
load_image_button = ctk.CTkButton(app, text="Cargar Imagen", command=image_app.load_image)
load_image_button.pack(pady=10)

# Campo de entrada para iteraciones de suavizado
blur_label = ctk.CTkLabel(app, text="Iteraciones de Suavizado (DE):")
blur_label.pack(pady=5)
blur_entry = ctk.CTkEntry(app, textvariable=image_app.blur_iterations)
blur_entry.pack(pady=5)

# Campo de entrada para iteraciones de afilado
sharp_label = ctk.CTkLabel(app, text="Iteraciones de Afilado (Tabu):")
sharp_label.pack(pady=5)
sharp_entry = ctk.CTkEntry(app, textvariable=image_app.sharp_iterations)
sharp_entry.pack(pady=5)

# Botón de "Continuar" que valida los campos y procesa la imagen si es válido
button_continue = ctk.CTkButton(app, text="Continuar", command=image_app.on_continue)
button_continue.pack(pady=20)

# Ejecutar la ventana principal
app.mainloop()