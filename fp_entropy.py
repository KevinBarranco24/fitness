import os
import cv2
import numpy as np

class Fp:
    @staticmethod
    def calculate_entropy(channel):
        if channel.dtype != np.uint8:
            channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        hist = hist[hist > 0]

        return -np.sum(hist * np.log2(hist))

    @staticmethod
    def value(x):     
        y = round(x[0], 8)
        z = round(x[1], 8)
        w = round(x[2], 8)

        print("Valores iniciales: ", x)
        print("Gain_Comp: ", y)
        print("Grad_Mbb: ", z)
        print("Num_Bands_Mbb: ", int(w))

        # Ejecutar script externo con parámetros
        path = f"image_stitching/main.py Imagenes/EDIFICIO_Z --gain-sigma-n {y} --mbb-sigma {z} --num-bands {int(w)}"
        os.system('python ' + path)

        path2 = r"Imagenes/EDIFICIO_Z/results"
        files_names = os.listdir(path2)

        entropies = []

        for file_name in files_names:
            image_path = os.path.join(path2, file_name)
            print(image_path)
            image = cv2.imread(image_path, 1)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            entropy_r = Fp.calculate_entropy(image_rgb[:, :, 0])
            entropy_g = Fp.calculate_entropy(image_rgb[:, :, 1])
            entropy_b = Fp.calculate_entropy(image_rgb[:, :, 2])
            entropy_avg = np.sqrt((entropy_r**2 + entropy_g**2 + entropy_b**2) / 3)

            entropies.append(entropy_avg)

        if len(entropies) == 0:
            print("No se pudieron analizar imágenes. Reintentando...")
            return Fp.value(x)

        Fit = np.mean(entropies)
        print(f"Entropía promedio (Fitness): {Fit:.4f}")

        return Fit
