#Hyper-entangled photon source testbench
#Author: Alexander Greenwood, Jackson Russett
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

from heps_static_sim import *

plt.rcParams['text.usetex'] =   True
plt.rcParams["font.family"] =   "serif"
plt.rcParams["font.size"] =     "14"

if __name__ == "__main__":
    N=30
    matrix_size = 50
    '''
        We assume full bandwidth of the PPSF biphoton spectrum to be 10THz in the case of no filter.
    '''

    #powerimbalan_st(1.2e12,0.5*np.sqrt(0.5),np.sqrt(0.5),N,matrix_size)
    #powerimbalan_st(6.0e12,0.5*np.sqrt(0.5),np.sqrt(0.5),N,matrix_size)

    #length_angle_st(1.2e12,-0.1,0.1,-20,20,N,matrix_size)
    #length_angle_st(6.0e12,-0.1,0.1,-20,20,N,matrix_size)
    #length_angle_dt(0.4e12, 0.8e12,-0.1,0.1,-20,20,N,matrix_size)

    #length_angle_inft(0.6e12,-0.1,0.1,-20,20,matrix_size)
    #length_angle_inft(1.8e12,-0.1,0.1,-20,20,matrix_size)
    #length_angle_inft(3.0e12,-0.1,0.1,-20,20,matrix_size)

    dws = np.linspace(0.2,2.4,50)*1e12
    #length_angle_inft_slices(dws, -0.1, 0.1, 0, 90, size=matrix_size)
    #length_angle_st_slices(dws, -0.1, 0.1, 0, 90, N=N, size=matrix_size)
    length_angle_dt_slices(0.4e12, dws, -0.1, 0.1, 0, 90, N=N, size=matrix_size)
    
    # Display the plots
    plt.show()  