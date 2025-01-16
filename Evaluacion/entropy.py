
from skimage import io
from skimage.measure import shannon_entropy #NECESITAMOS INSTALAR SCIKIT IMAGE SI NO NO FUNCIONA :V
import cv2
import numpy as np

# Cargar la imagen en formato RGB
image = io.imread("../Imagenes/results/pano_0.jpg")

# Calcular la entropía para cada canal
entropy_r = shannon_entropy(image[:, :, 0])  # Canal rojo
entropy_g = shannon_entropy(image[:, :, 1])  # Canal verde
entropy_b = shannon_entropy(image[:, :, 2])  # Canal azul

# Promedio de entropías
entropy_avg = np.sqrt((entropy_r**2 + entropy_g**2 + entropy_b**2) / 3)

print(f"Entropía por canal: R={entropy_r}, G={entropy_g}, B={entropy_b}")
print(f"Entropía promedio: {entropy_avg}")