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

# A Sample Cost Function
def value(x):
    y = 0
    for i in range(0,len(x)):
        y = x[i]
    print("Valor x: ", x)
    print("valor y: ", y)
    path = r"D:/Users/chevi/Documents/STITCHING_2007/image_stitching_2/main.py D:/Users/chevi/Documents/imagenes --gain-sigma-n "+str(y)
    os.system('python '+ path)
    path2 = r"D:/Users/chevi/Documents/imagenes/results"
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
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.resize(image, (1000, 700), interpolation=cv2.INTER_CUBIC)
            
        #cv2.imshow("Image", image)
        
        #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    result = image.astype(np.uint8)
    I_R = []
    I_G = []
    I_B = []

    I_R = result.copy()
    I_G = result.copy()
    I_B = result.copy()

    I_B[:,:,1]=0
    I_B[:,:,2]=0

    I_G[:,:,0]=0
    I_G[:,:,2]=0

    I_R[:,:,0]=0
    I_R[:,:,1]=0

    #M_R
    #Obtener filas y columnas
    R = np.array(I_R.shape[0])
    L = np.array(I_R.shape[1])
    intenR = np.zeros(1)
    CR = 1/(R*L)

    for i in range(1,R):
        for j in range(1,L):
            intenR = intenR + I_R[i][j][2]
    M_R = CR*intenR;

    #M_G
    #Obtener filas y columnas
    R = np.array(I_G.shape[0])
    L = np.array(I_G.shape[1])
    intenG = np.zeros(1)
    CG = 1/(R*L)

    for i in range(1,R):
        for j in range(1,L):
            intenG = intenG + I_G[i][j][1]    
    M_G = CG*intenG;

    #M_B
    #Obtener filas y columnas
    R = np.array(I_B.shape[0])
    L = np.array(I_B.shape[1])
    intenB = np.zeros(1)
    CB = 1/(R*L)

    for i in range(1,R):
        for j in range(1,L):
            intenB = intenB + I_B[i][j][0]
    M_B = CB*intenB;

    #Mean Value
    Fit = (M_R + M_G + M_B)/3;

    print("M_R ", M_R)
    print("M_G ", M_G)
    print("M_B ", M_B)
    print("Mean Value ", Fit)
    return Fit

# Define Optimization Problem
problem = {
        'CostFunction': value, 
        'nVar': 1, 
        'VarMin': 1,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
        'VarMax': 255,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
    }

# Running PSO
pso.tic()
print('Running PSO ...')
gbest, pop = pso.PSO(problem, MaxIter = 20, PopSize = 10, c1 = 1.5, c2 = 2, w = 1, wdamp = 0.995)
print()
pso.toc()
print()

# Final Result
print('Global Best:')
print(gbest)
print()
