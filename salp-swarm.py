############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Salp Swarm Algorithm

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Salp_Swarm_Algorithm, File: Python-MH-Salp Swarm Algorithm.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Salp_Swarm_Algorithm>

############################################################################

# Required Libraries
import numpy  as np
import math
import random
import os
from ssa import SsaF
import pandas as pd

# Function
def target_function():
    return

# Function: Initialize Variables
def initial_position(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((swarm_size, len(min_values) + 1))
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

# Function: Initialize Food Position
def food_position(dimension = 3, target_function = target_function):
    food = np.zeros((1, dimension+1))
    for j in range(0, dimension):
        food[0,j] = 0.0
    food[0,-1] = target_function(food[0,0:food.shape[1]-1])
    return food

# Function: Updtade Food Position by Fitness
def update_food(position, food):
    for i in range(0, position.shape[0]):
        if (food[0,-1] > position[i,-1]):
            for j in range(0, position.shape[1]):
                food[0,j] = position[i,j]
    return food

# Function: Updtade Position
def update_position(position, food, c1 = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    for i in range(0, position.shape[0]):
        if (i <= position.shape[0]/2):
            for j in range (0, len(min_values)):
                c2 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                c3 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                if (c3 >= 0.5): #c3 < 0.5
                    position[i,j] = np.clip((food[0,j] + c1*((max_values[j] - min_values[j])*c2 + min_values[j])), min_values[j],max_values[j])
                else:
                    position[i,j] = np.clip((food[0,j] - c1*((max_values[j] - min_values[j])*c2 + min_values[j])), min_values[j],max_values[j])                       
        elif (i > position.shape[0]/2 and i < position.shape[0] + 1):
            for j in range (0, len(min_values)):
                position[i,j] = np.clip(((position[i - 1,j] + position[i,j])/2), min_values[j],max_values[j])             
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])         
    return position

# SSA Function
def salp_swarm_algorithm(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function):    
    count    = 0
    position = initial_position(swarm_size = swarm_size, min_values = min_values, max_values = max_values, target_function = target_function)
    food     = food_position(dimension = len(min_values), target_function = target_function)
    while (count <= iterations):     
        print("Iteration = ", count, " f(x) = ", food[0,-1])
        data = {'Iteración ': [count], 'Best Cost': [food[0,-1]], 'Elapset time (seg)': [0]}
        try:
            df_e = pd.read_csv('best_ssa.csv')
            os.remove('best_ssa.csv')
            print(data)
            df = pd.DataFrame(data)
            n_df = pd.concat([df_e, df], ignore_index=True)
            n_df.to_csv('best_ssa.csv', index=False)
        except:
            print('Create new DF')
            print(data)
            df = pd.DataFrame(data)
            df.to_csv('best_ssa.csv', index=False) 
        c1       = 2*math.exp(-(4*(count/iterations))**2)
        food     = update_food(position, food)        
        position = update_position(position, food, c1 = c1, min_values = min_values, max_values = max_values, target_function = target_function)  
        count    = count + 1 
    print(food)    
    return food

######################## Part 1 - Usage ####################################

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
        data = {'Iteración ': [0], 'Best Cost': [0], 'Elapset time (seg)': [dt]}
        try:
            df_e = pd.read_csv('best_ssa.csv')
            os.remove('best_ssa.csv')
            print(data)
            df = pd.DataFrame(data)
            n_df = pd.concat([df_e, df], ignore_index=True)
            n_df.to_csv('best_ssa.csv', index=False)
        except:
            print('Create new DF')
            print(data)
            df = pd.DataFrame(data)
            df.to_csv('best_ssa.csv', index=False)
    else:
        print('Start time not set. You should call tic before toc.')



# Function to be Minimized (Six Hump Camel Back). Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def six_hump_camel_back(variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

##ssa = salp_swarm_algorithm(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 100, target_function = six_hump_camel_back)

# Function to be Minimized (Rosenbrocks Valley). Solution ->  f(x) = 0; xi = 1
def rosenbrocks_valley(variables_values = [0,0]):
    func_value = 0
    last_x = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + (100 * math.pow((variables_values[i] - math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)
    return func_value

##ssa = salp_swarm_algorithm(swarm_size = 15, min_values = [-5,-5], max_values = [5,5], iterations = 200, target_function = rosenbrocks_valley)
tic()
ssa = salp_swarm_algorithm(swarm_size = 5, min_values = [1,1, 0], max_values = [5,5, 10], iterations = 2, target_function = SsaF.value)
toc()