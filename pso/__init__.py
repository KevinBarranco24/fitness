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

import numpy as np
import os
import pandas as pd

# Particle Swarm Optimization
def PSO(problem, MaxIter = 100, PopSize = 100, c1 = 1.4962, c2 = 1.4962, w = 0.7298, wdamp = 1.0):

    # Empty Particle Template
    empty_particle = {
        'position': None, 
        'velocity': None, 
        'cost': None, 
        'best_position': None, 
        'best_cost': None, 
    }

    # Extract Problem Info
    CostFunction = problem['CostFunction']
    VarMin = problem['VarMin']
    VarMax = problem['VarMax']
    nVar = problem['nVar']

    # Initialize Global Best
    gbest = {'position': None, 'cost': np.inf}

    # Create Initial Population
    pop = []
    for i in range(0, PopSize):
        pop.append(empty_particle.copy())
        pop[i]['position'] = np.random.uniform(VarMin, VarMax, nVar)
        pop[i]['velocity'] = np.zeros(nVar)
        pop[i]['cost'] = CostFunction(pop[i]['position'])
        if(pop[i]['cost'] == 0 or pop[i]['cost'] == None ):
           pop[i]['cost'] = CostFunction(pop[i]['position']) 
        pop[i]['best_position'] = pop[i]['position'].copy()
        pop[i]['best_cost'] = pop[i]['cost']
        
        if pop[i]['best_cost'] < gbest['cost']:
            gbest['position'] = pop[i]['best_position'].copy()
            gbest['cost'] = pop[i]['best_cost']
    
    # PSO Loop
    for it in range(0, MaxIter):
        for i in range(0, PopSize):
            
            pop[i]['velocity'] = w*pop[i]['velocity'] \
                + c1*np.random.rand(nVar)*(pop[i]['best_position'] - pop[i]['position']) \
                + c2*np.random.rand(nVar)*(gbest['position'] - pop[i]['position'])

            pop[i]['position'] += pop[i]['velocity']
            pop[i]['position'] = np.maximum(pop[i]['position'], VarMin)
            pop[i]['position'] = np.minimum(pop[i]['position'], VarMax)

            pop[i]['cost'] = CostFunction(pop[i]['position'])
            if(pop[i]['cost'] == 0 or pop[i]['cost'] == None ):
                pop[i]['cost'] = CostFunction(pop[i]['position']) 
            
            if pop[i]['cost'] < pop[i]['best_cost']:
                pop[i]['best_position'] = pop[i]['position'].copy()
                pop[i]['best_cost'] = pop[i]['cost']

                if pop[i]['best_cost'] < gbest['cost']:
                    gbest['position'] = pop[i]['best_position'].copy()
                    gbest['cost'] = pop[i]['best_cost']

        w *= wdamp
        print('Iteration {}: Best Cost = {}'.format(it, gbest['cost']))
        data = {'Iteración ': [it], 'Best Cost': [gbest['cost']], 'Values': [gbest['position']], 'Elapset time (seg)': [0]}
        try:
            df_e = pd.read_csv('best_pso.csv')
            os.remove('best_pso.csv')
            print(data)
            df = pd.DataFrame(data)
            n_df = pd.concat([df_e, df], ignore_index=True)
            n_df.to_csv('best_pso.csv', index=False)
        except:
            print('Create new DF')
            print(data)
            df = pd.DataFrame(data)
            df.to_csv('best_pso.csv', index=False)

    return gbest, pop

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
            df_e = pd.read_csv('best_pso.csv')
            os.remove('best_pso.csv')
            print(data)
            df = pd.DataFrame(data)
            n_df = pd.concat([df_e, df], ignore_index=True)
            n_df.to_csv('best_pso.csv', index=False)
        except:
            print('Create new DF')
            print(data)
            df = pd.DataFrame(data)
            df.to_csv('best_pso.csv', index=False)
    else:
        print('Start time not set. You should call tic before toc.')
