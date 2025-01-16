############################################################################
# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Salp Swarm Algorithm
############################################################################

# Required Libraries
import numpy as np
import math
import random
import os
import pandas as pd
from ssa import SsaF

# Function
def target_function():
    return

# Function: Initialize Variables
def initial_position(swarm_size=5, min_values=[-5, -5], max_values=[5, 5], target_function=target_function):
    position = np.zeros((swarm_size, len(min_values) + 1))
    for i in range(swarm_size):
        for j in range(len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, :-1])
    return position

# Function: Initialize Food Position
def food_position(dimension=3, target_function=target_function):
    food = np.zeros((1, dimension + 1))
    food[0, -1] = float('inf')  # Initialize with a large value for minimization
    return food

# Function: Update Food Position by Fitness
def update_food(position, food):
    for i in range(position.shape[0]):
        if position[i, -1] < food[0, -1]:  # Minimization
            food[0, :-1] = position[i, :-1]
            food[0, -1] = position[i, -1]
    return food

# Function: Update Position
def update_position(position, food, c1=1, min_values=[-5, -5], max_values=[5, 5], target_function=target_function):
    for i in range(position.shape[0]):
        if i <= position.shape[0] // 2:  # Leaders
            for j in range(len(min_values)):
                c2 = random.random()
                c3 = random.random()
                if c3 >= 0.5:
                    position[i, j] = np.clip(
                        food[0, j] + c1 * ((max_values[j] - min_values[j]) * c2 + min_values[j]),
                        min_values[j],
                        max_values[j],
                    )
                else:
                    position[i, j] = np.clip(
                        food[0, j] - c1 * ((max_values[j] - min_values[j]) * c2 + min_values[j]),
                        min_values[j],
                        max_values[j],
                    )
        else:  # Followers
            for j in range(len(min_values)):
                position[i, j] = np.clip(
                    (position[i - 1, j] + position[i, j]) / 2, min_values[j], max_values[j]
                )
        position[i, -1] = target_function(position[i, :-1])
    return position

# SSA Function
def salp_swarm_algorithm(swarm_size=5, min_values=[-5, -5], max_values=[5, 5], iterations=50, target_function=target_function):
    position = initial_position(
        swarm_size=swarm_size, min_values=min_values, max_values=max_values, target_function=target_function
    )
    food = food_position(dimension=len(min_values), target_function=target_function)

    best_positions = []  # To track the best positions during optimization

    for count in range(iterations):
        food = update_food(position, food)
        position = update_position(
            position, food, c1=2 * math.exp(-(4 * (count / iterations))**2), min_values=min_values, max_values=max_values, target_function=target_function
        )
        best_positions.append((count, food[0, :-1].tolist(), food[0, -1]))
        print(f"Iteration {count}: Best Fitness = {food[0, -1]:.4f}")

    print("\nBest Overall Position:", food[0, :-1])
    print("Best Overall Fitness:", food[0, -1])
    return food, best_positions

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

if __name__ == "__main__":
    swarm_size = 5
    min_values = [1,1, 0]
    max_values = [5,5, 10]
    iterations = 2

    tic()
    best_food, best_positions = salp_swarm_algorithm(
        swarm_size=swarm_size,
        min_values=min_values,
        max_values=max_values,
        iterations=iterations,
        target_function=SsaF.value,
    )
    toc()
    # Save best positions to a file
    pd.DataFrame(best_positions, columns=["Iteration", "Position", "Fitness"]).to_csv("best_ssa.csv", index=False)
