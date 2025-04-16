import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread("../Imagenes/results/pano_0.jpg").astype(np.float32)

def calculate_gradient_opencv(channel):
    # Calcular gradientes con Sobel (más rápido y preciso que np.diff)
    grad_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=1)

    # Magnitud del gradiente
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # Promedio del gradiente
    return cv2.mean(grad_magnitude)[0]

# Calcular gradiente promedio por canal
gradient_r = calculate_gradient_opencv(image[:, :, 2])  # Canal rojo
gradient_g = calculate_gradient_opencv(image[:, :, 1])  # Canal verde
gradient_b = calculate_gradient_opencv(image[:, :, 0])  # Canal azul

# Promedio total
gradient_avg = (gradient_r + gradient_g + gradient_b) / 3

print(f"Gradiente promedio por canal: R={gradient_r:.4f}, G={gradient_g:.4f}, B={gradient_b:.4f}")
print(f"Gradiente promedio total: {gradient_avg:.4f}")
