#Hyper-entangled photon source dynamic simulations
#Author: Jackson Russett
import numpy as np
import time
import matplotlib.pyplot as plt
import QuantumTomography as qKLib
from qutip import *

from heps_dynamic_sim import *

## Specify some numpy and plotting settings
np.set_printoptions(suppress=True, precision=4)
plt.rcParams['text.usetex'] =   True
plt.rcParams["font.family"] =   "serif"
plt.rcParams["font.size"] =     "14"

def main():
    '''
    We assume full bandwidth of the PPSF biphoton spectrum to be 10THz in the case of no filter.
    
    Useful definitions
        # power split ratio, R_ps = I1 / (I1 + I2)
        # brightness amplitude factor, p ~ sqrt(R_ps)

    Some observations...
        # period T=400s, R_ps ampl=23%   based on PumpInputSpliced+NewFoam_PBSout_3.fig and assuming C is 0.5 -> range is then 27% to 73%
        ## raw counts fluctuate 10% from 400-450Hz
        ## PA counts fuctuate 40% from 130-180Hz

    Other notes
        After running state estimation, fval above the number of tomographic measurements indicates poor agreement of state with data
          from https://quantumtomo.web.illinois.edu/Doc/StateTomography_Matrix  
    ''' 

    # Initialize Tomography Object
    tomo = qKLib.Tomography()
    tomo.importConf('conf.txt')

    start_time = time.time()

    single_setting_example(tomo)
    #(trial_data, phis) = mc_phase(num_trials=1000, add_poisson_noise=False)
    #(trial_data, phis, static_settings, fluc_settings) = mc_phase(tomo, num_trials=1000, add_poisson_noise=True)
    #np.savez('results/Dynamic2p4THz_T400_A10p_C50p_MismSrc_PNoise.npz', td=trial_data, phis=phis, ss=static_settings, fs=fluc_settings)
    
    # Record the end time
    end_time = time.time()
    # Calculate the execution time
    execution_time = end_time - start_time
    print(f'Execution time: {execution_time/60.0} minutes')

    #display_mc_singleparam_results(trial_data, phis, 'phase (rad)')

    plt.show()


if __name__ == "__main__":
    main()