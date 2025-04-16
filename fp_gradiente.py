import os
import cv2
import numpy as np

class Fp:
    @staticmethod
    def calculate_gradient_opencv(channel):
        grad_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=1)
        grad_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=1)
        grad_magnitude = cv2.magnitude(grad_x, grad_y)
        return cv2.mean(grad_magnitude)[0]

    @staticmethod
    def value(x):     
        y = round(x[0], 8)
        z = round(x[1], 8)
        w = round(x[2], 8)
        
        print("Valores iniciales: ", x)
        print("Gain_Comp: ", y)
        print("Grad_Mbb: ", z)
        print("Num_Bands_Mbb: ", int(w))

        path = f"image_stitching/main.py Imagenes/EDIFICIO_Z --gain-sigma-n {y} --mbb-sigma {z} --num-bands {int(w)}"
        os.system('python ' + path)

        path2 = r"Imagenes/EDIFICIO_Z/results"
        files_names = os.listdir(path2)

        gradients = []

        for file_name in files_names:
            image_path = os.path.join(path2, file_name)
            print(image_path)
            image = cv2.imread(image_path, 1)
            if image is None:
                continue

            image = image.astype(np.float32)
            gradient_r = Fp.calculate_gradient_opencv(image[:, :, 2])
            gradient_g = Fp.calculate_gradient_opencv(image[:, :, 1])
            gradient_b = Fp.calculate_gradient_opencv(image[:, :, 0])
            gradient_avg = (gradient_r + gradient_g + gradient_b) / 3
            gradients.append(gradient_avg)

        if len(gradients) == 0:
            print("No se encontraron imágenes válidas. Reintentando...")
            return Fp.value(x)

        Fit = np.mean(gradients)
        print(f"Gradiente promedio (Fitness): {Fit:.4f}")

        return Fit
