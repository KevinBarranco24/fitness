import pso
import os
import cv2
import numpy as np

def calculate_entropy(channel):
    # Asegurar tipo uint8
    if channel.dtype != np.uint8:
        channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Histograma normalizado
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]

    # Entropía de Shannon
    return -np.sum(hist * np.log2(hist))

# Función de evaluación (fitness)
def value(x):
    y = round(x[0], 8)
    z = round(x[1], 8)
    w = round(x[2], 8)

    print("Valores iniciales: ", x)
    print("Gain_Comp: ", y)
    print("Grad_Mbb: ", z)
    print("Num_Bands_Mbb: ", int(w))

    # Ejecutar el script de stitching con parámetros actuales
    path = f"image_stitching/main.py Imagenes/EDIFICIO_Z --gain-sigma-n {y} --mbb-sigma {z} --num-bands {int(w)}"
    os.system('python ' + path)

    path2 = "Imagenes/EDIFICIO_Z/results"
    files_names = os.listdir(path2)

    entropies = []
    for file_name in files_names:
        image_path = os.path.join(path2, file_name)
        print("Analizando:", image_path)

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        entropy_r = calculate_entropy(image_rgb[:, :, 0])
        entropy_g = calculate_entropy(image_rgb[:, :, 1])
        entropy_b = calculate_entropy(image_rgb[:, :, 2])
        entropy_avg = np.sqrt((entropy_r**2 + entropy_g**2 + entropy_b**2) / 3)
        entropies.append(entropy_avg)

    # Evaluar la entropía promedio de todas las imágenes resultantes
    if len(entropies) == 0:
        print("No se pudieron analizar imágenes.")
        return value(x)  # Reintentar

    Fit = np.mean(entropies)
    print(f"Entropía promedio (Fitness): {Fit:.4f}")
    return Fit

# Definición del problema de optimización
problem = {
    'CostFunction': value,
    'nVar': 3,
    'VarMin': 1,
    'VarMax': 10,
}

# Ejecutar PSO
pso.tic()
print('Ejecutando PSO ...')
gbest, pop = pso.PSO(problem, MaxIter=5, PopSize=5, c1=1.5, c2=2, w=1, wdamp=0.995)
pso.toc()

# Resultado final
print('\nGlobal Best:')
print(gbest)
