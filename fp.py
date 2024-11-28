import os
import cv2
import numpy as np
import pandas as pd

class Fp:
    def value(x):     
    
        y = round(x[0],8)
        z = round(x[1],8)
        w = round(x[2],8)
        
        print("Valores_iniciales: ", x)
        print("Gain_Comp: ", y)
        print("Grad_Mbb: ", z)
        print("Num_Bands_Mbb: ", int(w))

        #PATH 1 ES LA RUTA DONDE SE EJECUTA EL CODIGO DE IMAGE_STITCHING Y LA RUTA DONDE SE ENCUENTRAN LAS FOTOS QUE SE VAN A UTILIZAR 
        path = r"image_stitching\main.py Imagenes --gain-sigma-n "+str(y)+" --mbb-sigma "+str(z)+" --num-bands "+ str(int(w))
        os.system('python '+ path)
        path2 = r"Imagenes/results"
        files_names = os.listdir(path2)
        for file_name in files_names:
            #print(file_name)
            """ 
            if file_name.split(".")[-1] not in ["jpeg", "png"]:
                continue
            """  
            image_path = path2 + "/" + file_name
            print(image_path)
            image = cv2.imread(image_path, 1)
            if image is None:
                continue
            
        #split con opencv
            b,g,r = cv2.split(image) 
        #suma de todos los pixeles
            #M_B
            R = np.array(b.shape[0])
            L = np.array(b.shape[1])
            CB = 1/(R*L)
            intenB = cv2.sumElems(b)[0]
            M_B = CB*intenB
            #M_G
            R = np.array(g.shape[0])
            L = np.array(g.shape[1])
            CG = 1/(R*L)
            intenG = cv2.sumElems(g)[0]
            M_G = CG*intenG
            #M_R
            R = np.array(r.shape[0])
            L = np.array(r.shape[1])
            CR = 1/(R*L)
            intenR = cv2.sumElems(r)[0]
            M_R = CR*intenR

        #Mean Value
        Fit = (M_R + M_G + M_B)/3;
        Fit
        
        return Fit