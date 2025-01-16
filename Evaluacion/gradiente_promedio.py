import cv2
import numpy as np

# Cargar la imagen (AQUI SE MODIFICA LA RUTA A DONDE SE ENCUENTREN LAS IMAGENES A EVALUAR)
image = cv2.imread("../Imagenes/results/pano_0.jpg")

# Convertir a flotante para mayor precisión
image = image.astype(np.float32)


def calculate_gradient(channel):
    # Calcular las diferencias en las direcciones horizontal y vertical
    grad_x = np.diff(channel, axis=1)  # Diferencia horizontal (Resta una columna)
    grad_y = np.diff(channel, axis=0)  # Diferencia vertical (Resta una fila)

    # Recortar para que las dimensiones coincidan
    grad_x = grad_x[:-1, :]  # Eliminar última fila para coincidir con grad_y
    grad_y = grad_y[:, :-1]  # Eliminar última columna para coincidir con grad_x

    # Calcular la magnitud del gradiente
    grad_magnitude = np.sqrt((grad_x**2 + grad_y**2) / 2)

    # Promedio del gradiente
    return np.mean(grad_magnitude)

# Calcular el gradiente promedio por canal
gradient_r = calculate_gradient(image[:, :, 2])  # Canal rojo
gradient_g = calculate_gradient(image[:, :, 1])  # Canal verde
gradient_b = calculate_gradient(image[:, :, 0])  # Canal azul

# Promedio total
gradient_avg = (gradient_r + gradient_g + gradient_b) / 3

print(f"Gradiente promedio por canal: R={gradient_r}, G={gradient_g}, B={gradient_b}")
print(f"Gradiente promedio total: {gradient_avg}")