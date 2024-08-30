#Hyper-entangled photon source simulation update August 2024
#Author: Jackson Russett
import numpy as np
import time
import matplotlib.pyplot as plt
from qutip import *

from heps_static_sim import *

plt.rcParams['text.usetex'] =   True
plt.rcParams["font.family"] =   "serif"
plt.rcParams["font.size"] =     "14"

if __name__ == "__main__":
    N=30
    matrix_size = 100
    '''
        We assume full bandwidth of the PPSF biphoton spectrum to be 10THz in the case of no filter.
    '''

    start_time = time.time()

    #powerimbalan_st(1.2e12,0.5*np.sqrt(0.5),np.sqrt(0.5),N,matrix_size)
    #powerimbalan_st(6.0e12,0.5*np.sqrt(0.5),np.sqrt(0.5),N,matrix_size)

    #length_angle_st(1.2e12,-0.1,0.1,-20,20,N,matrix_size)
    #length_angle_st(6.0e12,-0.1,0.1,-20,20,N,matrix_size)
    #length_angle_dt(0.4e12, 0.8e12,-0.1,0.1,-20,20,N,matrix_size)

    #length_angle_inft(0.6e12,-0.02,0.02,-10,10,matrix_size)
    #length_angle_inft(1.8e12,-0.02,0.02,-10,10,matrix_size)
    fig, ax1, ax2 = length_angle_inft(3.0e12,-0.02,0.02,-10,10,matrix_size)
    ax1.plot(np.linspace(-0.5,matrix_size-0.5,10), np.linspace(-0.5,matrix_size-0.5,10), color='red', linewidth=2)
    ax2.plot(np.linspace(-0.5,matrix_size-0.5,10), np.linspace(-0.5,matrix_size-0.5,10), color='red', linewidth=2)

    dws = np.linspace(1.2,6.0,100)*1e12
    fig, ax1, ax2 = length_angle_inft_slices(dws, -0.02,0.02,-10,10, size=matrix_size)
    idx = np.argmin(abs(dws-3.0e12))
    ax1.plot(np.linspace(-0.5,matrix_size-0.5,10), idx*np.ones((10,)), color='red', linewidth=2)
    ax2.plot(np.linspace(-0.5,matrix_size-0.5,10), idx*np.ones((10,)), color='red', linewidth=2)
    length_angle_st_slices(dws, 0, 0.02, 0, 10, N=N, size=matrix_size)
    dws = np.linspace(0.2,3.0,100)*1e12
    length_angle_dt_slices(0.4e12, dws, -0.1, 0.1, 0, 90, N=N, size=matrix_size)
    

    # Record the end time
    end_time = time.time()
    # Calculate the execution time
    execution_time = end_time - start_time
    print(f'Execution time: {execution_time/60.0} minutes')

    # Display the plots
    plt.show()  