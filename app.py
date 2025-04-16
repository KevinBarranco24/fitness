"""

Copyright (c) 2017, Mostapha Kalami Heris & Yarpiz (www.yarpiz.com)
All rights reserved. Please read the "LICENSE" file for usage terms.
__________________________________________________________________________

Project Code: YPEA127
Project Title: Implementation of Particle Swarm Optimization in Python
Publisher: Yarpiz (www.yarpiz.com)

Developer: Mostapha Kalami Heris (Member of Yarpiz Team)

Cite as:
Mostapha Kalami Heris, Particle Swarm Optimization (PSO) in Python (URL: https://yarpiz.com/463/ypea127-pso-in-python), Yarpiz, 2017.

Contact Info: sm.kalami@gmail.com, info@yarpiz.com

"""

import pso
import os
import cv2
import numpy as np
import pandas as pd

# A Sample Cost Function
def value(x):     
    
    y = round(x[0],8)
    z = round(x[1],8)
    w = round(x[2],8)
    
    
    print("Valores_iniciales: ", x)
    print("Gain_Comp: ", y)
    print("Grad_Mbb: ", z)
    print("Num_Bands_Mbb: ", int(w))

    #PATH 1 ES LA RUTA DONDE SE EJECUTA EL CODIGO DE IMAGE_STITCHING Y LA RUTA DONDE SE ENCUENTRAN LAS FOTOS QUE SE VAN A UTILIZAR 
    path = r"image_stitching/main.py Imagenes/EDIFICIO_Z --gain-sigma-n "+str(y)+" --mbb-sigma "+str(z)+" --num-bands "+ str(int(w))
    os.system('python '+ path)
    path2 = r"Imagenes/EDIFICIO_Z/results"
    files_names = os.listdir(path2)
    for file_name in files_names:
        
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

    if Fit == 0.0:
        print("Fit es 0.0, recalculando...")
        return value(x)
    
    print("Fitness: ", Fit)

    return Fit

# Define Optimization Problem
problem = {
        'CostFunction': value, 
        'nVar': 3, 
        'VarMin': 1,   # LIMITE MINIMO SIGMA N GAIN COMPENSATION
        'VarMax': 10, # LIMITE MAXIMO SIGMA N GAIN COMPENSATION
    }

# Running PSO

pso.tic()
print('Running PSO ...')
gbest, pop = pso.PSO(problem, MaxIter = 5, PopSize = 35, c1 = 1.5, c2 = 2, w = 1, wdamp = 0.995)
print()
pso.toc()
print()

# Final Result
print('Global Best:')
print(gbest)
print()
