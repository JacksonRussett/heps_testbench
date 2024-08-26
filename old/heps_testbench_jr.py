#Hyper-entangled photon source testbench
#Author: Alexander Greenwood, Jackson Russett
import numpy as np
import matplotlib.pyplot as plt
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
c = 2.997925 * 10 ** (8) #m/s, speed of light

h_pol_ket = basis(2,0)
v_pol_ket = basis(2,1)
i_freq_ket = basis(2,0)
s_freq_ket = basis(2,1)
hs_ket = tensor(h_pol_ket,s_freq_ket)
hi_ket = tensor(h_pol_ket,i_freq_ket)
vs_ket = tensor(v_pol_ket,s_freq_ket)
vi_ket = tensor(v_pol_ket,i_freq_ket)
hshi_ket = tensor(hs_ket,hi_ket)
hihs_ket = tensor(hi_ket,hs_ket)
hsvi_ket = tensor(hs_ket,vi_ket)
hivs_ket = tensor(hi_ket,vs_ket)
vsvi_ket = tensor(vs_ket,vi_ket)
vivs_ket = tensor(vi_ket,vs_ket)
vshi_ket = tensor(vs_ket,hi_ket)
vihs_ket = tensor(vi_ket,hs_ket)


def get_he_state(w_s,w_i,l1=1,l2=1,l1_p=1,l2_p=1,p=0.7071,a1=0,a2=0):
    '''Define our hyperentangled state at ports 3 and 4 given lengths inside the sagnac loop 
    for compensation L_1,L_1' (l1, l1_p respectively),etc.
    
    '''
    basis_ws = basis(2,0)
    basis_wi = basis(2,1)
    basis_h = basis(2,0)
    basis_v = basis(2,1)

    #Definition of constants:
    M = 1.297564e-12     # unit = 10^-12 s/m to avoid calculation with extremely large or small numbers
                    # correspondingly, \Delta n should be in the unit of rad * THz (or rad * 10^12/s)
    e = np.e
    kc = 5927434.262 # SI unit
    # Assuming \Delta\omega = 2 pi * 2 THz
    Mdw = M * 2 * np.pi * 2.0e12

    # Assuming l_2 +l_2 - l_1 - l_1' = 0, we don't need the k_c term

    alpha = lambda w : p*(np.cos(a1)**2 * np.exp(Mdw*l1*1j) - np.sin(a1)**2 * np.exp(-Mdw*l1*1j))*np.exp(Mdw*l1_p*1j)
    beta =  lambda w : p*(np.cos(a1)**2 * np.exp(-Mdw*l1*1j) - np.sin(a1)**2 * np.exp(Mdw*l1*1j))*np.exp(-Mdw*l1_p*1j)
    gamma = lambda w : np.sqrt(1-p**2)*(np.cos(a2)**2 * np.exp(-Mdw*l2*1j) - np.sin(a2)**2 * np.exp(Mdw*l2*1j))*np.exp(-Mdw*l2_p*1j)*np.exp(2*kc*(l2+l2_p-l1-l1_p)*1j)
    lambd = lambda w : np.sqrt(1-p**2)*(np.cos(a2)**2 * np.exp(Mdw*l2*1j) - np.sin(a2)**2 * np.exp(-Mdw*l2*1j))*np.exp(Mdw*l2_p*1j)*np.exp(2*kc*(l2+l2_p-l1-l1_p)*1j)
    
    w = 1
    #print(alpha(w))#,beta(w),gamma(w),lambd(w))
    he_state = 1/np.sqrt(2)*(alpha(w)*tensor(tensor(tensor(basis_h,basis_ws),basis_v),basis_wi) + \
                beta(w)*tensor(tensor(tensor(basis_h,basis_wi),basis_v),basis_ws) + \
                gamma(w)*tensor(tensor(tensor(basis_v,basis_ws),basis_h),basis_wi) + \
                lambd(w)*tensor(tensor(tensor(basis_v,basis_wi),basis_h),basis_ws))


    return he_state

def get_hep_state(settings):
    '''Direct implimentation of eqn 7.9 from Changjia's thesis
    '''
    wvl_range = settings['filter_range']
    l1 = settings['l1']
    l2 = settings['l2']
    l1_p = settings['l1_p']
    l2_p = settings['l2_p']
    p = settings['p']
    a1 = settings['a1']
    a2 = settings['a2']

    biref = 5e-4 # birefringence (unitless)

    n_v = 1.5
    k_v = lambda w : w / c*n_v

    n_h = n_v + biref
    k_h = lambda w : w / c*n_h

    lambda_deg = 1556e-9
    w_deg = 2*np.pi*c/lambda_deg
    w_s = lambda w : w_deg + w
    w_i = lambda w : w_deg - w

    #Definition of constants:
    M = 1.297564e-12     # unit = 10^-12 s/m to avoid calculation with extremely large or small numbers
                    # correspondingly, \Delta n should be in the unit of rad * THz (or rad * 10^12/s)

    kc = 5927434.262 # SI unit
    # Definition of constants in case needed
    beatlength = 4 * 10 ** (-3)
    e = np.e

    n_gv = 1.4682 #assume fast axis group index is roughly same as SMF28
    v_gh = (n_gv/c + lambda_deg/(c*beatlength))**(-1)
    v_gv = c / n_gv
    kh_wc = kc + np.pi / beatlength
    kv_wc = kc - np.pi / beatlength

    Mdw = lambda dw: M * 2 * np.pi * dw

    alpha = lambda w : p*(np.cos(a1)**2 * np.exp(Mdw(w)*l1*1j) - np.sin(a1)**2 * np.exp(-Mdw(w)*l1*1j))*np.exp(Mdw(w)*l1_p*1j)
    beta =  lambda w : p*(np.cos(a1)**2 * np.exp(-Mdw(w)*l1*1j) - np.sin(a1)**2 * np.exp(Mdw(w)*l1*1j))*np.exp(-Mdw(w)*l1_p*1j)
    gamma = lambda w : np.sqrt(1-p**2)*(np.cos(a2)**2 * np.exp(-Mdw(w)*l2*1j) - np.sin(a2)**2 * np.exp(Mdw(w)*l2*1j))*np.exp(-Mdw(w)*l2_p*1j)*np.exp(2*kc*(l2+l2_p-l1-l1_p)*1j)
    lambd = lambda w : np.sqrt(1-p**2)*(np.cos(a2)**2 * np.exp(Mdw(w)*l2*1j) - np.sin(a2)**2 * np.exp(-Mdw(w)*l2*1j))*np.exp(Mdw(w)*l2_p*1j)*np.exp(2*kc*(l2+l2_p-l1-l1_p)*1j)

    #print(alpha(wvl_range))

    A = sum(np.abs(alpha(wvl_range))**2 + np.abs(beta(wvl_range))**2)
    B = sum(alpha(wvl_range)*np.conjugate(gamma(wvl_range)) + beta(wvl_range)*np.conjugate(lambd(wvl_range)))
    C = sum(np.conjugate(alpha(wvl_range))*gamma(wvl_range) + np.conjugate(beta(wvl_range))*lambd(wvl_range))
    D = sum(np.abs(gamma(wvl_range))**2 + np.abs(lambd(wvl_range))**2)
    #print(A, B, C, D)
    he_state = Qobj(np.array([
        [0, 0, 0, 0],
        [0, A, B, 0],
        [0, C, D, 0],
        [0, 0, 0, 0]
    ]),dims=[[2,2],[2,2]])

    return he_state.unit()

def define_state(dw, theta_1,theta_2,l_1,l_1p,l_2,l_2p):
    #Definition of constants:
    M = 1.297564e-12     # unit = 10^-12 s/m to avoid calculation with extremely large or small numbers
                    # correspondingly, \Delta n should be in the unit of rad * THz (or rad * 10^12/s)
    e = np.e
    kc = 5927434.262 # SI unit
    # Assuming \Delta\omega = 2 pi * 2 THz
    Mdw = M * 2 * np.pi * dw
    p = 1/np.sqrt(2) #Amplitude placeholder

    # Assuming l_2 +l_2 - l_1 - l_1' = 0, we don't need the k_c term

    non_id1 = (np.cos(theta_1)**2*e**(Mdw*l_1*1j) - np.sin(theta_1)**2*e**(-Mdw*l_1*1j))*e**(Mdw*l_1p*1j)
    # # This just corresponds to nonidealities of first term in output state

    non_id2 = (np.cos(theta_1)**2*e**(-Mdw*l_1*1j) - np.sin(theta_1)**2*e**(Mdw*l_1*1j))*e**(-Mdw*l_1p*1j)

    non_id3 = (np.cos(theta_2)**2*e**(-Mdw*l_2*1j) - np.sin(theta_2)**2*e**(Mdw*l_2*1j))*e**(-Mdw*l_2p*1j)*e**(2*kc*(l_2+l_2p-l_1-l_1p)*1j)

    non_id4 = (np.cos(theta_2)**2*e**(Mdw*l_2*1j) - np.sin(theta_2)**2*e**(-Mdw*l_2*1j))*e**(Mdw*l_2p*1j)*e**(2*kc*(l_2+l_2p-l_1-l_1p)*1j)

    term_1 = p*hsvi_ket*non_id1
    term_2 = p*hivs_ket*non_id2
    term_3 = np.sqrt(1-p**2)*vshi_ket*non_id3
    term_4 = np.sqrt(1-p**2)*vihs_ket*non_id4

    #print(np.abs(non_id1), np.abs(non_id2), np.abs(non_id3), np.abs(non_id4))

    return 1/(np.sqrt(2))*(term_1 + term_2 + term_3 + term_4)

def single_setting_example(N=100):
    single_settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.020,
        'l2':           1.000,
        'l2_p':         0.980,                                  # lengths have similar effect as Calvin's thesis
        'a1':           90.0 * np.pi/180.0,                      # !!! angle has less effect than in Calvin's thesis
        'a2':           90.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),                            # power imbalance has similar effect to Calvin's thesis
        'filter_range': np.linspace(0.0e12,3.0e12,N)*2*np.pi      # assume max PPSF bandwidth of 10THz?
    }   
    
    print('ALEX', concurrence(define_state(2.0e12, 
                                           single_settings['a1'],
                                           single_settings['a2'],
                                           single_settings['l1'],
                                           single_settings['l1_p'],
                                           single_settings['l2'],
                                           single_settings['l2_p']).ptrace([0,2])))


    #print(single_settings['filter_range'])

    # density matrix sum method
    he_l = []
    he_normsqr = []
    w_deg = 2*np.pi*c/single_settings['lambda_deg']
    Dw = single_settings['filter_range'][1] - single_settings['filter_range'][0]
    for i in range(0,N):
        w_s = w_deg + single_settings['filter_range'][i]
        w_i = w_deg - single_settings['filter_range'][i]
        he_state = define_state(single_settings['filter_range'][i]/(2*np.pi),single_settings['a1'],single_settings['a2'],single_settings['l1'],single_settings['l1_p'],single_settings['l2'],single_settings['l2_p'])#single_settings['p']
        he_l.append(he_state)
        #print('MSTE',concurrence(he_state.ptrace([0,2])), he_state.norm())
    
    #print(he_l)
    #print(sum(he_normsqr).norm()) 
    he_sum = (sum(he_l))/N#/np.sqrt(sum(he_normsqr).norm())#np.sqrt(sum(he_normsqr).norm())
    #print(he_sum)
    print('MSUM',concurrence(he_sum.ptrace([0,2])), he_sum.norm())

def main():
    N=50
    single_setting_example(N)
    
    settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.000,
        'l2':           1.000,
        'l2_p':         1.000,                                  # lengths have similar effect as Calvin's thesis
        'a1':           90.0 * np.pi/180.0,                      # !!! angle has less effect than in Calvin's thesis
        'a2':           90.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),                            # power imbalance has similar effect to Calvin's thesis
        'filter_range': np.linspace(0.0e12,0.5*10e12,N)*2*np.pi      # assume max PPSF bandwidth of 10THz?
    }
    #print(settings['filter_range'])

    # Define the size of the matrix
    size = 30

    def conc_length_mismatch(m1, m2):
        settings['l1_p'] = settings['l1'] + m1
        settings['l2_p'] = settings['l2'] + m2
        # density matrix sum method
        he_l = []
        w_deg = 2*np.pi*c/settings['lambda_deg']
        for i in range(0,N):
            w_s = w_deg + settings['filter_range'][i]
            w_i = w_deg - settings['filter_range'][i]
            he_state = define_state(settings['filter_range'][i]/(2*np.pi),settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
            he_l.append(he_state)
            #print('MSTE',concurrence(he_state.ptrace([0,2])), he_state.norm())
    
        he_sum = (sum(he_l))/N
        return concurrence(he_sum.ptrace([0,2]))
    
    def conc_angle_mismatch(t1, t2):
        settings['a1'] = (90+t1) * np.pi/180.0
        settings['a2'] = (90+t2) * np.pi/180.0
        # density matrix sum method
        he_l = []
        w_deg = 2*np.pi*c/settings['lambda_deg']
        for i in range(0,N):
            w_s = w_deg + settings['filter_range'][i]
            w_i = w_deg - settings['filter_range'][i]
            he_state = define_state(settings['filter_range'][i]/(2*np.pi),settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
            he_l.append(he_state)
            #print('MSTE',concurrence(he_state.ptrace([0,2])), he_state.norm())
    
        he_sum = (sum(he_l))/N
        return concurrence(he_sum.ptrace([0,2]))

    # Generate the matrix using the concurrence function
    mm1 = np.linspace(-0.1,0.1,size)
    mm2 = np.linspace(-0.1,0.1,size)
    matrix1 = np.zeros((size, size))
    for j, m2 in enumerate(mm2):
        for i, m1 in enumerate(mm1):
            matrix1[j, i] = conc_length_mismatch(m1, m2)

    settings['l1_p'] = settings['l1']
    settings['l2_p'] = settings['l2']

    # Generate the matrix using the concurrence function
    tt1 = np.linspace(-20,20,size)
    tt2 = np.linspace(-20,20,size)
    matrix2 = np.zeros((size, size))
    for j, t2 in enumerate(tt2):
        for i, t1 in enumerate(tt1):
            matrix2[j, i] = conc_angle_mismatch(t1, t2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
    im1 = ax1.imshow(matrix1, cmap='viridis', interpolation='nearest')

    # Step 3: Set x-ticks and y-ticks
    # Generate tick positions and labels
    tick_positions = np.linspace(0, size - 1, num=5)  # 5 ticks for simplicity
    xtick_labels = np.linspace(min(mm1), max(mm1), num=5)  
    ytick_labels = np.linspace(min(mm2), max(mm2), num=5)

    ax1.set_xticks(tick_positions, labels=np.round(xtick_labels, 2))
    ax1.set_yticks(tick_positions, labels=np.round(ytick_labels, 2))
    ax1.invert_yaxis()

    # Step 4: Add a colorbar
    fig.colorbar(im1, ax=ax1, label='Concurrence')

    # Step 5: Add labels and title
    ax1.set_xlabel('m1')
    ax1.set_ylabel('m2')
    ax1.set_title('Effect of Length Mismatch')

    im2 = ax2.imshow(matrix2, cmap='viridis', interpolation='nearest')
    tick_positions = np.linspace(0, size - 1, num=5)
    xtick_labels = np.linspace(min(tt1), max(tt1), num=5)  
    ytick_labels = np.linspace(min(tt2), max(tt2), num=5)
    ax2.set_xticks(tick_positions, labels=np.round(xtick_labels, 2))
    ax2.set_yticks(tick_positions, labels=np.round(ytick_labels, 2))
    ax2.invert_yaxis()
    fig.colorbar(im2, ax=ax2, label='Concurrence')
    ax2.set_xlabel('t1')
    ax2.set_ylabel('t2')
    ax2.set_title('Effect of Angular Mismatch')

    # Step 6: Display the plot
    plt.show()  



if __name__ == "__main__":
    main()