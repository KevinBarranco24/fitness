import cv2
import numpy as np

def calculate_entropy(channel):
    # Convertir a uint8 si no lo está
    if channel.dtype != np.uint8:
        channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Calcular histograma (256 niveles de gris)
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()  # Normalizar a probabilidades

    # Evitar log(0) eliminando ceros
    hist = hist[hist > 0]

    # Entropía de Shannon
    return -np.sum(hist * np.log2(hist))

# Leer imagen en BGR (OpenCV) y convertir a RGB
image_bgr = cv2.imread("../Imagenes/results/pano_0.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Calcular entropía por canal
entropy_r = calculate_entropy(image_rgb[:, :, 0])  # Rojo
entropy_g = calculate_entropy(image_rgb[:, :, 1])  # Verde
entropy_b = calculate_entropy(image_rgb[:, :, 2])  # Azul

# Promedio de entropías (con raíz cuadrada del promedio cuadrático)
entropy_avg = np.sqrt((entropy_r**2 + entropy_g**2 + entropy_b**2) / 3)

print(f"Entropía por canal: R={entropy_r:.4f}, G={entropy_g:.4f}, B={entropy_b:.4f}")
print(f"Entropía promedio: {entropy_avg:.4f}")
