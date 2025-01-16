import cv2
import numpy as np

# Cargar la imagen (AQUI SE MODIFICA LA RUTA A DONDE SE ENCUENTREN LAS IMAGENES A EVALUAR)
image = cv2.imread("../Imagenes/results/pano_0.jpg")

# Convertir a flotante para mayor precisión
image = image.astype(np.float32)

# Calcular el brillo promedio por canal (BGR en OpenCV)
height, width, _ = image.shape
brightness_r = np.mean(image[:, :, 2])  # Canal rojo
brightness_g = np.mean(image[:, :, 1])  # Canal verde
brightness_b = np.mean(image[:, :, 0])  # Canal azul

# Promedio de los canales
brightness_avg = (brightness_r + brightness_g + brightness_b) / 3

print(f"Brillo promedio por canal: R={brightness_r}, G={brightness_g}, B={brightness_b}")
print(f"Brillo promedio total: {brightness_avg}")