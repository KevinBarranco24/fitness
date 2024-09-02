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
    
    """
    df = pd.DataFrame(columns=['y_sigma_n','z_mbb_sigma','w_num_bands_mbb','Mean_Value'])
    df._append({'y_sigma_n': y},ignore_index=True) 
    df._append({'y_sigma_n': z},ignore_index=True) 
    df._append({'y_sigma_n': w},ignore_index=True)
    """ 
    print("Valor x: ", x)
    print("valor y: ", y)
    print("valor y: ", z)
    print("valor y: ", int(w))
    #PATH 1 ES LA RUTA DONDE SE EJECUTA EL CODIGO DE IMAGE_STITCHING Y LA RUTA DONDE SE ENCUENTRAN LAS FOTOS QUE SE VAN A UTILIZAR 
    path = r"D:/Users/chevi/Documents/STITCHING_2007/image_stitching_2/main.py D:/Users/chevi/Documents/imagenes --gain-sigma-n "+str(y)+" --mbb-sigma "+str(z)+" --num-bands "+ str(int(w))
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

    '''
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
    M_B = CB*intenB;'''
    
    #Mean Value
    Fit = (M_R + M_G + M_B)/3;
    '''
    data = {'y_sigma_n': [y],'z_mbb_sigma': [z],'w_num_bands_mbb': [w],'Mean_Value': [Fit],'Best_Cost': [0]}
    #df._append({'Mean_Value': Fit},ignore_index=True)
    print("M_R ", M_R)
    print("M_G ", M_G)
    print("M_B ", M_B)
    print("Mean Value ", Fit)
    try:
        df_e = pd.read_csv('log_pso.csv')
        os.remove('log_pso.csv')
        print(data)
        df = pd.DataFrame(data)
        n_df = pd.concat([df_e, df], ignore_index=True)
        n_df.to_csv('log_pso.csv', index=False)

    except:
        print('Create new DF')
        print(data)
        df = pd.DataFrame(data)
        df.to_csv('log_pso.csv', index=False)
    '''
    return Fit

# Define Optimization Problem
problem = {
        'CostFunction': value, 
        'nVar': 3, 
        'VarMin': 1,   # LIMITE MINIMO SIGMA N GAIN COMPENSATION
        'VarMax': 10, # LIMITE MAXIMO SIGMA N GAIN COMPENSATION
        #'VarMin2': 2,   # LIMITE MINIMO NUMERO DE BANDAS MULTIBANDBLENDING
        #'VarMax2': 10,  # LIMITE MAXIMO NUMERO DE BANDAS MULTIBANDBLENDING
        #'VarMin3': 1,   # LIMITE MINIMO PARA SIGMA DE MULTIBANDBLENDING
        #'VarMax3': 5,   # LIMITE MAXIMO PARA SIGMA DE MULTIBANDBLENDING
    }

# Running PSO

pso.tic()
print('Running PSO ...')
gbest, pop = pso.PSO(problem, MaxIter = 5, PopSize = 5, c1 = 1.5, c2 = 2, w = 1, wdamp = 0.995)
print()
pso.toc()
print()
'''
data = {'y_sigma_n': [0],'z_mbb_sigma': [0],'w_num_bands_mbb': [0],'Mean_Value': [0],'Best_Cost': [gbest['cost']]}
df_e = pd.read_csv('log_pso.csv')
os.remove('log_pso.csv')
print(data)
df = pd.DataFrame(data)
n_df = pd.concat([df_e, df], ignore_index=True)
n_df.to_csv('log_pso.csv', index=False)
'''
# Final Result
print('Global Best:')
print(gbest)
print()
