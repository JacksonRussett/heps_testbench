import numpy as np
from qutip import *

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

def define_state(dw, theta_1,theta_2,l_1,l_1p,l_2,l_2p,p=1/np.sqrt(2)):
    #Definition of constants:
    M = 1.297564e-12     # unit = 10^-12 s/m to avoid calculation with extremely large or small numbers
                    # correspondingly, \Delta n should be in the unit of rad * THz (or rad * 10^12/s)
    e = np.e
    kc = 5927434.262 # SI unit
    # Assuming \Delta\omega = 2 pi * 2 THz
    Mdw = M * 2 * np.pi * dw
    #p =  #Amplitude placeholder

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
