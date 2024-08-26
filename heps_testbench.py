#Hyper-entangled photon source testbench
#Author: Alexander Greenwood
import numpy as np
from qutip import *


'''

Configuration of the setup that we wish to simulate.
________x______
|              |   _________
|______        |  |
|_PPSF_|       PBS
|              |  |_________
|_______x______|


'''

########DEFINE GLOBAL CONSTANTS############
c=3e8 #m/s, speed of light


def get_he_state(w_s,w_i,l1=1,l2=1,l1_p=1,l2_p=1):
    '''Define our hyperentangled state at ports 3 and 4 given lengths inside the sagnac loop 
    for compensation L_1,L_1' (l1, l1_p respectively),etc.
    
    '''
    biref = 5e-4 # birefringence (unitless)

    

    n_v = 1.5
    k_v = lambda w : w / c*n_v

    n_h = n_v + biref
    k_h = lambda w : w / c*n_h

    basis_ws = basis(2,0)
    basis_wi = basis(2,1)

    basis_h = basis(2,0)
    basis_v = basis(2,1)

    p = np.sqrt(1-0.5)

    alpha = lambda w : p*np.exp(((k_h(w_s) + k_v(w_i))*l1 + (k_v(w_s) + k_h(w_i))*l1_p)*1j)#p*(np.cos(a1)**2 * np.exp((k_h(w_s) + k_v(w_i))*l1*1j) - np.sin(a1)**2 * np.exp((k_v(w_s) + k_h(w_i))*l1*1j))*np.exp((k_h(w_s) + k_v(w_i))*l1_p*1j)
    beta =  lambda w : p*np.exp(((k_v(w_s) + k_h(w_i))*l1 + (k_h(w_s) + k_v(w_i))*l1_p)*1j)#p*(np.cos(a1)**2 * np.exp((k_v(w_s) + k_h(w_i))*l1*1j) - np.sin(a1)**2 * np.exp((k_h(w_s) + k_v(w_i))*l1*1j))*np.exp((k_v(w_s) + k_h(w_i))*l1_p*1j)
    gamma = lambda w : np.sqrt(1-p**2)*np.exp(((k_h(w_s) + k_v(w_i))*l2 + (k_v(w_s) + k_h(w_i))*l2_p)*1j)#np.sqrt(1-p**2)*(np.cos(a2)**2 * np.exp((k_v(w_s) + k_h(w_i))*l2*1j) - np.sin(a2)**2 * np.exp((k_h(w_s) + k_v(w_i))*l2*1j))*np.exp((k_v(w_s) + k_h(w_i))*l2_p*1j)
    lambd = lambda w : np.sqrt(1-p**2)*np.exp(((k_v(w_s) + k_h(w_i))*l2 + (k_h(w_s) + k_v(w_i))*l2_p)*1j)#np.sqrt(1-p**2)*(np.cos(a2)**2 * np.exp((k_h(w_s) + k_v(w_i))*l2*1j) - np.sin(a2)**2 * np.exp((k_v(w_s) + k_h(w_i))*l2*1j))*np.exp((k_h(w_s) + k_v(w_i))*l2_p*1j)
    
    w=1
    #print(alpha(w))#,beta(w),gamma(w),lambd(w))
    he_state = 1/np.sqrt(2)*(alpha(w)*tensor(tensor(tensor(basis_h,basis_ws),basis_v),basis_wi) + \
                             beta(w)*tensor(tensor(tensor(basis_h,basis_wi),basis_v),basis_ws) + \
                             gamma(w)*tensor(tensor(tensor(basis_v,basis_wi),basis_h),basis_ws) + \
                             lambd(w)*tensor(tensor(tensor(basis_v,basis_ws),basis_h),basis_wi))


    return he_state


def main():
    lambda_deg = 1556e-9
    w_deg = c/lambda_deg*2*np.pi
    w_s = w_deg + 10e10*2*np.pi
    w_i = w_deg - 10e10*2*np.pi
    N=100

    he_l = []
    detuning_range = np.linspace(0,0.5*6.0e12,N)

    for i in range(0,N):
        w_s = w_deg + detuning_range[i]*2*np.pi
        w_i = w_deg - detuning_range[i]*2*np.pi
        he_state = get_he_state(w_s,w_i,l1=1.00,l2=1.00).unit()
        he_l.append(he_state)
        #print(concurrence(he_state.ptrace([0,2])))
    
    he_sum = (sum(he_l)).unit()
    print(concurrence(he_sum.ptrace([0,2])))

if __name__ == "__main__":
    main()