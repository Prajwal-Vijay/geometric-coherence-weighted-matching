"""
This includes the main initialization of the states of evaders and pursuers.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize_scalar
import time
import Environment
import Pursuer
import Evader
import random

def main():
    plt.close('all')
    N=3 # Number of pursuers
    M=10 # Number of evaders
    t = 0.1 # Time step
    pursuer_positions = np.array([(0, 2, 12), (24, 2, 12), (12, 20, 12)])
    evader_positions = np.array([
            (18, 5, 12),
            (6, 15, 12),
            (4, 5, 12),
            (20, 15, 12),
            (2, 5, 12),
            (22, 5, 12),
            (8, 15, 12),
            (16, 15, 12),
            (10, 5, 12),
            (14, 5, 12),
        ]
    )
    pursuer_speeds = np.array([1.74, 1.48, 1.51])
    evader_speeds = np.array([1.01, 0.97, 1.0, 0.96, 0.99, 0.99, 0.98, 0.98, 1.03, 1.02])
    pursuers = []
    evaders = []

    for i in range(M):
        evaders.append(Evader.Evader(evader_positions[i], evader_speeds[i], i))
    for i in range(N):
        pursuers.append(Pursuer.Pursuer(pursuer_positions[i], pursuer_speeds[i], i))
    env = Environment.Environment(N, M, t, pursuers, evaders, strategy="nearest_single")
    win = env.check_initialization(True)
    print(win)
    env.plot_current_positions()
    # We are trying to avoid the condition B_ij >= 0 and alpha_ij >= 1, if this happens
    # we will not even run the simulation, because the evaders will endup winning irrespective of how
    # smartly the pursuers play.
    
    if win:
        print("Running simulation")
        win_result = env.obtain_trajectories()
        # print(win_result)
    else:
        print('Irrespecitive of how the pursuer plays, evaders end up winning. UNFAIR!')

if __name__ == "__main__":
    main()
