import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import messagebox
#from fp import Fp
#from fp_entropy import Fp
from fp_gradiente import Fp
import os
import pandas as pd

# Start Time for tic and tov functions
startTime_for_tictoc = 0

# Start measuring time elapsed
def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

# End mesuring time elapsed
def toc():
    import time, math
    if 'startTime_for_tictoc' in globals():
        dt = math.floor(100*(time.time() - startTime_for_tictoc))/100.
        print('Elapsed time is {} second(s).'.format(dt))
        data = {'Iteración ': [0], 'Best Cost': [0], 'Values': [0], 'Elapset time (seg)': [dt]}
        try:
            df_e = pd.read_csv('best_gwo.csv')
            os.remove('best_gwo.csv')
            print(data)
            df = pd.DataFrame(data)
            n_df = pd.concat([df_e, df], ignore_index=True)
            n_df.to_csv('best_gwo.csv', index=False)
        except:
            print('Create new DF')
            print(data)
            df = pd.DataFrame(data)
            df.to_csv('best_gwo.csv', index=False)
    else:
        print('Start time not set. You should call tic before toc.')

def initialization (PopSize,D,LB,UB):
    SS_Boundary = len(LB) if isinstance(UB,(list,np.ndarray)) else 1
    if SS_Boundary ==1:
        Positions = np.random.rand(PopSize,D)*(UB-LB)+LB
    else:
        Positions = np.zeros((PopSize,D))
        for i in range(D):
            Positions[:,i]=np.random.rand(PopSize)*(UB[i]-LB[i])+LB[i]
    return Positions

def GWO(PopSize,MaxT,LB,UB,D,Fobj):
    Alpha_Pos = np.zeros(D)
    Alpha_Fit = np.inf
    Beta_Pos = np.zeros(D)
    Beta_Fit = np.inf
    Delta_Pos = np.zeros(D)
    Delta_Fit = np.inf

    Positions = initialization(PopSize,D,UB,LB)
    Convergence_curve = np.zeros(MaxT)

    l = 0
    while l<MaxT:
        for i in range (Positions.shape[0]):
            BB_UB = Positions[i,:]>UB 
            BB_LB = Positions[i,:]<LB
            Positions[i,:] = (Positions[i,:]*(~(BB_UB+BB_LB)))+UB*BB_UB+LB*BB_LB
            Fitness = Fobj(Positions[i,:])
            if(Fitness == 0 or Fitness == None ):
                Fitness = Fobj(Positions[i,:])
            print("Fitness: ", Fitness)
            if Fitness<Alpha_Fit:
                Alpha_Fit=Fitness
                Alpha_Pos=Positions[i,:]

            if Fitness>Alpha_Fit and Fitness<Beta_Fit:
                Beta_Fit=Fitness
                Beta_Pos=Positions[i,:]
            
            if Fitness>Alpha_Fit and Fitness>Beta_Fit and Fitness<Delta_Fit:
                Delta_Fit=Fitness
                Delta_Pos=Positions[i,:]
        
        a = 2-1*(2/MaxT)
        for i in range (Positions.shape[0]):
            for j in range (Positions.shape[1]):
                r1=np.random.random()
                r2=np.random.random()

                A1 = 2*a*r1-a
                C1 = 2 * r2

                D_Alpha = abs(C1*Alpha_Pos[j]-Positions[i,j])
                X1 = Alpha_Pos[j]-A1*D_Alpha
                
                r1=np.random.random()
                r2=np.random.random()

                A2 = 2*a*r1-a
                C2=2*r2

                D_Beta = abs(C2*Beta_Pos[j]-Positions[i,j])
                X2 = Beta_Pos[j]-A2*D_Beta

                r1 = np.random.random()
                r2 = np.random.random()

                A3 = 2*a*r1-a
                C3 = 2*r2

                D_Delta = abs(C3 * Delta_Pos[j] - Positions[i,j])
                X3 = Delta_Pos[j] - A3 * D_Delta

                Positions[i,j] = (X1 + X2 + X3) / 3
            
        l += 1
        Convergence_curve[l - 1] = Alpha_Fit
        data = {'Iteración ': [l], 'Best Cost': [Alpha_Fit], 'Values': [Alpha_Pos], 'Elapset time (seg)': [0]}
        try:
            df_e = pd.read_csv('best_gwo.csv')
            os.remove('best_gwo.csv')
            print(data)
            df = pd.DataFrame(data)
            n_df = pd.concat([df_e, df], ignore_index=True)
            n_df.to_csv('best_gwo.csv', index=False)
        except:
            print('Create new DF')
            print(data)
            df = pd.DataFrame(data)
            df.to_csv('best_gwo.csv', index=False)
        
    return Alpha_Fit, Alpha_Pos, Convergence_curve

if __name__ == "__main__":
    def F1(x):
        return np.sum(x ** 2)

    Fun_name = Fp.value
    LB = 1
    UB = 10
    D = 3
    PopSize= 5
    MaxT = 5
    tic()
    bestfit, bestsol, convergence_curve = GWO(PopSize,MaxT,LB,UB,D,Fun_name)
    print("Best Fitness =", bestfit)
    print("Best Solution = ",bestsol)
    print()
    toc()
    print()

# Show the final result in a message box
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("GWO Result", f"Best Fitness: {bestfit}\nBest Solution: {bestsol}")

    plt.plot(convergence_curve)
    plt.xlabel("Iteraciones")
    plt.ylabel("Fitness")
    plt.title("GWO convergence curve")
    plt.grid(True)
    plt.show()
