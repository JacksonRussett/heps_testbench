from sys import *
from qutip import *
from scipy.stats import truncnorm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import pdb
#from basic_units import radians, degrees, cos


# plt.rc('xtick', labelsize=14) 
# plt.rc('ytick', labelsize=14) 
n_states = 2

#entangled states computation:

h_pol_ket = basis(n_states,0)
v_pol_ket = basis(n_states,1)
i_freq_ket = basis(n_states,0)
s_freq_ket = basis(n_states,1)
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

#Definition of constants:
M = 1.297564     # unit = 10^-12 s/m to avoid calculation with extremely large or small numbers
                 # correspondingly, \Delta n should be in the unit of rad * THz (or rad * 10^12/s)

kc = 5927434.262 # SI unit
# Definition of constants in case needed
c = 2.997925 * 10 ** (8)
wl = 1.556 * 10 ** (-6)      # degeneracy wavelength
beatlength = 4 * 10 ** (-3)
e = np.e

n_gv = 1.4682 #assume fast axis group index is roughly same as SMF28
v_gh = (n_gv/c + wl/(c*beatlength))**(-1)
v_gv = c / n_gv
kh_wc = kc + np.pi / beatlength
kv_wc = kc - np.pi / beatlength
dw = 3 * np.pi * 10**12 #  of rad/s 

kh_ws = kh_wc + dw / (v_gh)
kh_wi = kh_wc - dw / (v_gh)
kv_ws = kv_wc + dw/ (v_gv)
kv_wi = kv_wc - dw/ (v_gv)


'''
This code is only true for the case of complete matching of L_3/L_4. Two splices are used. 

def define_state(theta_1,theta_2,l_1,l_1p,l_2,l_2p):
    # Assuming \Delta\omega = 2 pi * 2 THz
    Mdw = M * 2 * np.pi * 2 
    p = 1/np.sqrt(2) #Amplitude placeholder

    # Assuming l_2 +l_2 - l_1 - l_1' = 0, we don't need the k_c term

    non_id1 = (np.cos(theta_1)**2*e**(Mdw*l_1*1j) - np.sin(theta_1)**2*e**(-Mdw*l_1*1j))*e**(Mdw*l_1p*1j)
    # This just corresponds to nonidealities of first term in output state

    non_id2 = (np.cos(theta_1)**2*e**(-Mdw*l_1*1j) - np.sin(theta_1)**2*e**(Mdw*l_1*1j))*e**(-Mdw*l_1p*1j)

    non_id3 = (np.cos(theta_2)**2*e**(-Mdw*l_2*1j) - np.sin(theta_2)**2*e**(Mdw*l_2*1j))*e**(-Mdw*l_2p*1j)

    non_id4 = (np.cos(theta_2)**2*e**(Mdw*l_2*1j) - np.sin(theta_2)**2*e**(-Mdw*l_2*1j))*e**(Mdw*l_2p*1j)

    term_1 = p*hsvi_ket*non_id1
    term_2 = p*hivs_ket*non_id2
    term_3 = np.sqrt(1-p**2)*vshi_ket*non_id3
    term_4 = np.sqrt(1-p**2)*vihs_ket*non_id4

    return 1/(np.sqrt(2))*(term_1 + term_2 + term_3 + term_4) 
'''


def define_state(theta_1,theta_2,l_1,l_1p,l_2,l_2p):
    # Assuming \Delta\omega = 2 pi * 2 THz
    Mdw = M * 2 * np.pi * 2
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

    print(non_id1)

    return 1/(np.sqrt(2))*(term_1 + term_2 + term_3 + term_4)



#Define stat for the case of 4 cross splices. 
def define_state_4x(theta_1,theta_2,theta_3, theta_4, l_1,l_1p,l_2,l_2p,l_3,l_3p,l_4,l_4p):
    # Assuming \Delta\omega = 2 pi * 2 THz
    Mdw = M * 2 * np.pi * 2
    p = 1/np.sqrt(2) #Amplitude placeholder
    
    #p = 1/np.sqrt(2) - 0.1
    # Assuming l_2 +l_2 - l_1 - l_1' = 0, we don't need the k_c term

    non_id1 = (np.cos(theta_1)**2*e**(Mdw*l_1*1j) - np.sin(theta_1)**2*e**(-Mdw*l_1*1j))*e**(Mdw*l_1p*1j)
    # This just corresponds to nonidealities of first term in output state

    non_id2 = (np.cos(theta_1)**2*e**(-Mdw*l_1*1j) - np.sin(theta_1)**2*e**(Mdw*l_1*1j))*e**(-Mdw*l_1p*1j)

    non_id3 = (np.cos(theta_2)**2*e**(-Mdw*l_2*1j) - np.sin(theta_2)**2*e**(Mdw*l_2*1j))*e**(-Mdw*l_2p*1j)*e**(2*kc*(l_2+l_2p-l_1-l_1p)*1j)

    non_id4 = (np.cos(theta_2)**2*e**(Mdw*l_2*1j) - np.sin(theta_2)**2*e**(-Mdw*l_2*1j))*e**(Mdw*l_2p*1j)*e**(2*kc*(l_2+l_2p-l_1-l_1p)*1j)

    term_1 = p*hsvi_ket*np.cos(theta_3)*np.cos(theta_4)*non_id1*e**((kh_ws*(l_3 + l_3p)+kv_wi*(l_4+l_4p))*1j) - np.sqrt(1-p**2)*np.sin(theta_3)*np.sin(theta_4)*hsvi_ket*non_id3*e**((kv_ws*l_3 + kh_ws*l_3p+kh_wi*l_4+kv_wi*l_4p)*1j)
    term_2 = p*hivs_ket*np.cos(theta_3)*np.cos(theta_4)*non_id2*e**((kh_wi*(l_3 + l_3p)+kv_ws*(l_4+l_4p))*1j) - np.sqrt(1-p**2)*np.sin(theta_3)*np.sin(theta_4)*hivs_ket*non_id4*e**((kv_wi*l_3 + kh_wi*l_3p+kh_ws*l_4+kv_ws*l_4p)*1j)
    term_3 = np.sqrt(1-p**2)*np.cos(theta_3)*np.cos(theta_4)*vshi_ket*non_id3*e**((kv_ws*(l_3 + l_3p)+kh_wi*(l_4+l_4p))*1j) - p*np.sin(theta_3)*np.sin(theta_4)*vshi_ket*non_id1*e**((kh_ws*l_3 + kv_ws*l_3p+kv_wi*l_4+kh_wi*l_4p)*1j)
    term_4 = np.sqrt(1-p**2)*np.cos(theta_3)*np.cos(theta_4)*vihs_ket*non_id4*e**((kv_wi*(l_3 + l_3p)+kh_ws*(l_4+l_4p))*1j) - p*np.sin(theta_3)*np.sin(theta_4)*vihs_ket*non_id2*e**((kh_wi*l_3 + kv_wi*l_3p+kv_ws*l_4+kh_ws*l_4p)*1j)
    
    #Do we disregard these because of postselection? vvvv
    # term_5 = -p*hshi_ket*np.cos(theta_3)*np.sin(theta_4)*non_id1*e**((kh_ws*(l_3 + l_3p)+kv_wi*l_4+kh_wi*l_4p)*1j) + np.sqrt(1-p**2)*np.sin(theta_3)*np.cos(theta_4)*hshi_ket*non_id3*e**((kv_ws*l_3 + kh_ws*l_3p+kh_wi*(l_4+l_4p))*1j)
    # term_6 = -p*hihs_ket*np.cos(theta_3)*np.sin(theta_4)*non_id2*e**((kh_wi*(l_3 + l_3p)+kv_ws*l_4+kh_ws*l_4p)*1j) + np.sqrt(1-p**2)*np.sin(theta_3)*np.cos(theta_4)*hihs_ket*non_id3*e**((kv_wi*l_3 + kh_wi*l_3p+kh_ws*(l_4+l_4p))*1j)
    # term_7 = np.sqrt(1-p**2)*np.cos(theta_3)*np.sin(theta_4)*vsvi_ket*non_id3*e**((kv_ws*(l_3 + l_3p)+kh_wi*l_4+kv_wi*l_4p)*1j) + p*np.sin(theta_3)*np.cos(theta_4)*vsvi_ket*non_id3*e**((kh_ws*l_3 + kv_ws*l_3p+kv_wi*(l_4+l_4p))*1j)
    # term_8 = np.sqrt(1-p**2)*np.cos(theta_3)*np.sin(theta_4)*vivs_ket*non_id4*e**((kv_wi*(l_3 + l_3p)+kh_ws*l_4+kv_ws*l_4p)*1j) + p*np.sin(theta_3)*np.cos(theta_4)*vivs_ket*non_id3*e**((kh_wi*l_3 + kv_wi*l_3p+kv_ws*(l_4+l_4p))*1j)
    
    term_5 = 0
    term_6 = 0
    term_7 = 0
    term_8 = 0
    
    return 1/(np.sqrt(2))*(term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8)

def theta_l_mismatch_2x():
    l_iterations = 1
    
    #It looks like we can compensate for L_1 with L_2 here???
    mismatch_factor_l1 = np.linspace(1.00,1.0,l_iterations)
    mismatch_factor_l2 = np.linspace(1.00,1,l_iterations)
    l_1 = 0.9 # this is just a placeholder. 
    l_1p = l_1*mismatch_factor_l1
    l_2 = l_1
    l_2p = l_2*mismatch_factor_l2
    l_3 = l_1
    l_3p = l_3
    l_4 = l_1
    l_4p = l_4
    
    N = 100
    theta_1_range = np.linspace((90-40)*np.pi/180,(90+40)*np.pi/180,N)
    theta_2_range = np.linspace((90-40)*np.pi/180,(90+40)*np.pi/180,N)
    #pdb.set_trace()
    concurrence_range = np.zeros((N,N))
    
    for k in range(0,len(l_1p)):    
        for l in range(0,len(l_2p)):
            for i in range(0,len(theta_1_range)):
                for j in range(0,len(theta_2_range)):
                    state_pbs = define_state(theta_1_range[i],theta_2_range[j],l_1,l_1p[k],l_2,l_2p[l])
                    #d_matrix = state_pbs * state_pbs.dag()
                    d_matrix = ket2dm(state_pbs)
                    #alternatively can use ket2dm
                    rho_pol = d_matrix.ptrace([0,2])
                    concurrence_range[i][j] = concurrence(rho_pol)
            plt.figure()
            plt.plot(theta_1_range,concurrence_range[:,1]) #Second index should align with theta_2 = pi/2
            plt.title(r'Concurrence vs. Angular Mismatch ($\theta_1$)' + '\n $L_1^\prime=$' + ('%.3f' % mismatch_factor_l1[k]) + r'$L_1$, $L_2^\prime=$'+ ('%.3f' % mismatch_factor_l2[l]) + r'$L_2$',fontsize=18)
            plt.ylabel("Concurrence", fontsize=18)
            plt.xlabel("Splice Angle (rad)", fontsize=18)
            
            #This was just taken from StackOverflow. Using this to format x axis so splice angle in units of pi rad
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
            
            ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
            #End of copied section
            plt.tick_params(axis='y', which='major', labelsize=18)
            plt.tick_params(axis='x', which='major', labelsize=20)
            ax.set_ylim([0.5,1])
            plt.figure()
            
            #Same as above. Copied from stack overflow to format x and y. 
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
            ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
            
            ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
            #End of copied section 
            
            cmap = plt.pcolormesh(theta_1_range,theta_2_range,concurrence_range)
            plt.clim(0.5,1) #Fixed colour scale.
            plt.colorbar(cmap)
            plt.title(r'Concurrence vs. Angular Mismatch ($\theta_1,\theta_2$)' + '\n$L_1^\prime=$' + ('%.3f' % mismatch_factor_l1[k]) + r'$L_1$, $L_2^\prime=$'+ ('%.3f' % mismatch_factor_l2[l]) + r'$L_2$',fontsize=18)
            plt.ylabel(r'$\theta_2$'+ " Splice Angle (rad)", fontsize=18)
            plt.xlabel(r'$\theta_1$'+ " Splice Angle (rad)", fontsize=18)
            plt.tick_params(axis='y', which='major', labelsize=20)
            plt.tick_params(axis='x', which='major', labelsize=20)
            concurrence_range = np.zeros([N,N])
    plt.show()
    return

def l_mismatch_2x():
    l_iterations = 100
    
    #It looks like we can compensate for L_1 with L_2 here???
    # mismatch_factor_l1 = np.linspace(0.9,1.1,l_iterations)
    # mismatch_factor_l2 = np.linspace(0.9, 1.1,l_iterations)
    # l_1 = 0.9 # this is just a placeholder. 
    # l_1p = l_1*mismatch_factor_l1
    # l_2 = l_1
    # l_2p = l_2*mismatch_factor_l2
    mismatch_l1 = np.linspace(-0.02,0.02,l_iterations)
    mismatch_l2 = np.linspace(-0.02,0.02,l_iterations)
    l_1 = 0.9 # this is just a placeholder. 
    l_1p = l_1 + mismatch_l1
    l_2 = l_1
    l_2p = l_2 + mismatch_l2
    
    theta_1 = np.pi / 2
    theta_2 = np.pi / 2
    concurrence_range = np.zeros([l_iterations,l_iterations])
    
    for k in range(0,len(l_1p)):    
        for l in range(0,len(l_2p)):
            state_pbs = define_state(theta_1,theta_2,l_1,l_1p[k],l_2,l_2p[l])
            d_matrix = state_pbs * state_pbs.dag()
            #alternatively can use ket2dm
            rho_pol = d_matrix.ptrace([0,2])
            concurrence_range[k][l] = concurrence(rho_pol)
    
    
    cmap = plt.pcolormesh(mismatch_l1*1000,mismatch_l2*1000,concurrence_range)
    #plt.clim(0.5,1) #Fixed colour scale.
    plt.colorbar(cmap)
    plt.title(r'Concurrence vs. Length Mismatch',fontsize=18)
    plt.ylabel(r'$\alpha_2$ (mm)', fontsize=18)
    plt.xlabel(r'$\alpha_1$ (mm)', fontsize=18)
    plt.tick_params(axis='y', which='major', labelsize=20)
    plt.tick_params(axis='x', which='major', labelsize=20)
    plt.show()
    return

def get_results_4x():
    sim_type = 'l1_3_theta_3_4'
    l_iterations = 4
    mismatch_factor_sl1 = np.linspace(1.00,1.08,l_iterations)
    mismatch_factor_sl2 = np.linspace(1.00,1.158,l_iterations)
    mismatch_factor = np.linspace(1.00,0.9,l_iterations)
    N = 20
    base_length = 0.965
    
    if (sim_type == 'l1_2_theta_1_2'):
        l_1 = base_length # This is the length used in the lab to the best of my knowledge.
        l_1p = l_1*mismatch_factor_sl1
        l_2 = l_1
        l_2p = l_2*mismatch_factor_sl2
        l_3 = l_1
        l_3p = l_3
        l_4 = l_1
        l_4p = l_4
        theta_1_range = np.linspace(0,np.pi,N)
        theta_2_range = np.linspace(0,np.pi,N)
        
    elif (sim_type == 'l1_2_theta_1_3'):
        l_1 = base_length 
        l_1p = l_1*mismatch_factor_sl1
        l_2 = l_1
        l_2p = l_2*mismatch_factor_sl2
        l_3 = l_1
        l_3p = l_3
        l_4 = l_1
        l_4p = l_4
        theta_1_range = np.linspace(0,np.pi,N)
        theta_3_range = np.linspace(0,np.pi,N)
        
    elif (sim_type == 'l1_2_theta_3_4'):
        l_1 = base_length
        l_1p = l_1*mismatch_factor_sl1
        l_2 = l_1
        l_2p = l_2*mismatch_factor_sl2
        l_3 = l_1
        l_3p = l_3
        l_4 = l_1
        l_4p = l_4
        theta_3_range = np.linspace(0,np.pi,N)
        theta_4_range = np.linspace(0,np.pi,N)
        
    elif (sim_type == 'l1_3_theta_1_2'):
        l_1 = base_length
        l_1p = l_1*mismatch_factor_sl1
        l_2 = l_1
        l_2p = l_2
        l_3 = l_1
        l_3p = l_3*mismatch_factor_sl2
        l_4 = l_1
        l_4p = l_4
        theta_1_range = np.linspace(0,np.pi,N)
        theta_2_range = np.linspace(0,np.pi,N)
        
    elif (sim_type == 'l1_3_theta_1_3'):
        l_1 = base_length
        l_1p = l_1*mismatch_factor_sl1
        l_2 = l_1
        l_2p = l_2
        l_3 = l_1
        l_3p = l_3*mismatch_factor_sl2
        l_4 = l_1
        l_4p = l_4
        theta_1_range = np.linspace(0,np.pi,N)
        theta_3_range = np.linspace(0,np.pi,N)
        
    elif (sim_type == 'l1_3_theta_3_4'):
        l_1 = base_length
        l_1p = l_1*mismatch_factor_sl1
        l_2 = l_1
        l_2p = l_2
        l_3 = l_1
        l_3p = l_3*mismatch_factor_sl2
        l_4 = l_1
        l_4p = l_4
        theta_3_range = np.linspace(0,np.pi,N)
        theta_4_range = np.linspace(0,np.pi,N)
   
    concurrence_range = np.zeros([N,N])
    
    #Defines the function call used depending on the type of simulation specified by user.
    
    #These function calls are contained in a dictionary to allow the user to change simulation
    #parameters with little hassle. 
    sim_dict = {'l1_2_theta_1_2':'define_state_4x(theta_1_range[i],theta_2_range[j], np.pi/2, np.pi/2,l_1,l_1p[k],l_2,l_2p[l],l_3,l_3p,l_4,l_4p)',
              'l1_2_theta_1_3':'define_state_4x(theta_1_range[i], np.pi/2, theta_3_range[j],np.pi/2,l_1,l_1p[k],l_2,l_2p[l],l_3,l_3p,l_4,l_4p)',
              'l1_2_theta_3_4':'define_state_4x(np.pi/2, np.pi/2, theta_3_range[i],theta_4_range[j],l_1,l_1p[k],l_2,l_2p[l],l_3,l_3p,l_4,l_4p)',
              'l1_3_theta_1_2':'define_state_4x(theta_1_range[i],theta_2_range[j], np.pi/2, np.pi/2,l_1,l_1p[k],l_2,l_2p,l_3,l_3p[l],l_4,l_4p)',
              'l1_3_theta_1_3':'define_state_4x(theta_1_range[i], np.pi/2, theta_3_range[j],np.pi/2,l_1,l_1p[k],l_2,l_2p,l_3,l_3p[l],l_4,l_4p)',
              'l1_3_theta_3_4':'define_state_4x(np.pi/2, np.pi/2, theta_3_range[i],theta_4_range[j],l_1,l_1p[k],l_2,l_2p,l_3,l_3p[l],l_4,l_4p)',
    }
    
    
    theta_sweep_1d_plot = {'l1_2_theta_1_2':'plt.plot(theta_1_range,concurrence_range[:,10])',
              'l1_2_theta_1_3':'plt.plot(theta_1_range,concurrence_range[:,10])',
              'l1_2_theta_3_4':'plt.plot(theta_3_range,concurrence_range[:,10])',
              'l1_3_theta_1_2':'plt.plot(theta_1_range,concurrence_range[:,10])',
              'l1_3_theta_1_3':'plt.plot(theta_1_range,concurrence_range[:,10])',
              'l1_3_theta_3_4':'plt.plot(theta_3_range,concurrence_range[:,10])' 
    }
    
    theta_sweep_1d_title = {'l1_2_theta_1_2':'plt.title(r\'Concurrence vs. Angular Mismatch ($\\theta_1$)\' + \'\\n $L_1^\prime=$\' + (\'\%.3f\' % mismatch_factor_sl1[k]) + r\'$L_1$, $L_2^\prime=$\'+ (\'%.3f\' % mismatch_factor_sl2[l]) + r\'$L_2$\',fontsize=18)',
              'l1_2_theta_1_3':'plt.title(r\'Concurrence vs. Angular Mismatch ($\\theta_1$)\' + \'\\n $L_1^\prime=$\' + (\'%.3f\' % mismatch_factor_sl1[k]) + r\'$L_1$, $L_2^\prime=$\'+ (\'%.3f\' % mismatch_factor_sl2[l]) + r\'$L_2$\',fontsize=18)',
              'l1_2_theta_3_4':'plt.title(r\'Concurrence vs. Angular Mismatch ($\\theta_3$)\' + \'\\n $L_1^\prime=$\' + (\'%.3f\' % mismatch_factor_sl1[k]) + r\'$L_1$, $L_2^\prime=$\'+ (\'%.3f\' % mismatch_factor_sl2[l]) + r\'$L_2$\',fontsize=18)',
              'l1_3_theta_1_2':'plt.title(r\'Concurrence vs. Angular Mismatch ($\\theta_1$)\' + \'\\n $L_1^\prime=$\' + (\'%.3f\' % mismatch_factor_sl1[k]) + r\'$L_1$, $L_3^\prime=$\'+ (\'%.3f\' % mismatch_factor_sl2[l]) + r\'$L_3$\',fontsize=18)',
              'l1_3_theta_1_3':'plt.title(r\'Concurrence vs. Angular Mismatch ($\\theta_1$)\' + \'\\n $L_1^\prime=$\' + (\'%.3f\' % mismatch_factor_sl1[k]) + r\'$L_1$, $L_3^\prime=$\'+ (\'%.3f\' % mismatch_factor_sl2[l]) + r\'$L_3$\',fontsize=18)',
              'l1_3_theta_3_4':'plt.title(r\'Concurrence vs. Angular Mismatch ($\\theta_3$)\' + \'\\n $L_1^\prime=$\' + (\'%.3f\' % mismatch_factor_sl1[k]) + r\'$L_1$, $L_3^\prime=$\'+ (\'%.3f\' % mismatch_factor_sl2[l]) + r\'$L_3$\',fontsize=18)' 
    }
    
    theta_sweep_2d_plot = {'l1_2_theta_1_2':'plt.pcolormesh(theta_1_range,theta_2_range,concurrence_range)',
              'l1_2_theta_1_3':'plt.pcolormesh(theta_1_range,theta_3_range,concurrence_range)',
              'l1_2_theta_3_4':'plt.pcolormesh(theta_3_range,theta_4_range,concurrence_range)',
              'l1_3_theta_1_2':'plt.pcolormesh(theta_1_range,theta_2_range,concurrence_range)',
              'l1_3_theta_1_3':'plt.pcolormesh(theta_1_range,theta_3_range,concurrence_range)',
              'l1_3_theta_3_4':'plt.pcolormesh(theta_3_range,theta_4_range,concurrence_range)' 
    }
    
    theta_sweep_2d_title = {'l1_2_theta_1_2':'plt.title(r\'Concurrence vs. Angular Mismatch ($\\theta_1,\\theta_2$)\' + \'\\n$L_1^\prime=$\' + (\'%.3f\' % mismatch_factor_sl1[k]) + r\'$L_1$, $L_2^\prime=$\'+ (\'%.3f\' % mismatch_factor_sl2[l]) + r\'$L_2$\',fontsize=18)',
              'l1_2_theta_1_3':'plt.title(r\'Concurrence vs. Angular Mismatch ($\\theta_1,\\theta_3$)\' + \'\\n$L_1^\prime=$\' + (\'%.3f\' % mismatch_factor_sl1[k]) + r\'$L_1$, $L_2^\prime=$\'+ (\'%.3f\' % mismatch_factor_sl2[l]) + r\'$L_2$\',fontsize=18)',
              'l1_2_theta_3_4':'plt.title(r\'Concurrence vs. Angular Mismatch ($\\theta_3,\\theta_4$)\' + \'\\n$L_1^\prime=$\' + (\'%.3f\' % mismatch_factor_sl1[k]) + r\'$L_1$, $L_2^\prime=$\'+ (\'%.3f\' % mismatch_factor_sl2[l]) + r\'$L_2$\',fontsize=18)',
              'l1_3_theta_1_2':'plt.title(r\'Concurrence vs. Angular Mismatch ($\\theta_1,\\theta_2$)\' + \'\\n$L_1^\prime=$\' + (\'%.3f\' % mismatch_factor_sl1[k]) + r\'$L_1$, $L_3^\prime=$\'+ (\'%.3f\' % mismatch_factor_sl2[l]) + r\'$L_3$\',fontsize=18)',
              'l1_3_theta_1_3':'plt.title(r\'Concurrence vs. Angular Mismatch ($\\theta_1,\\theta_3$)\' + \'\\n$L_1^\prime=$\' + (\'%.3f\' % mismatch_factor_sl1[k]) + r\'$L_1$, $L_3^\prime=$\'+ (\'%.3f\' % mismatch_factor_sl2[l]) + r\'$L_3$\',fontsize=18)',
              'l1_3_theta_3_4':'plt.title(r\'Concurrence vs. Angular Mismatch ($\\theta_3,\\theta_4$)\' + \'\\n$L_1^\prime=$\' + (\'%.3f\' % mismatch_factor_sl1[k]) + r\'$L_1$, $L_3^\prime=$\'+ (\'%.3f\' % mismatch_factor_sl2[l]) + r\'$L_3$\',fontsize=18)' 
    }
    
    theta_sweep_2d_axes = {'l1_2_theta_1_2':'plt.xlabel(r\'$\\theta_1$\'+ " Splice Angle (rad)",fontsize=18)\nplt.ylabel(r\'$\\theta_2$\'+ " Splice Angle (rad)",fontsize=18)',
              'l1_2_theta_1_3':'plt.xlabel(r\'$\\theta_1$\'+ " Splice Angle (rad)",fontsize=18)\nplt.ylabel(r\'$\\theta_3$\'+ " Splice Angle (rad)",fontsize=18)',
              'l1_2_theta_3_4':'plt.xlabel(r\'$\\theta_3$\'+ " Splice Angle (rad)",fontsize=18)\nplt.ylabel(r\'$\\theta_4$\'+ " Splice Angle (rad)",fontsize=18)',
              'l1_3_theta_1_2':'plt.xlabel(r\'$\\theta_1$\'+ " Splice Angle (rad)",fontsize=18)\nplt.ylabel(r\'$\\theta_2$\'+ " Splice Angle (rad)",fontsize=18)',
              'l1_3_theta_1_3':'plt.xlabel(r\'$\\theta_1$\'+ " Splice Angle (rad)",fontsize=18)\nplt.ylabel(r\'$\\theta_3$\'+ " Splice Angle (rad)",fontsize=18)',
              'l1_3_theta_3_4':'plt.xlabel(r\'$\\theta_3$\'+ " Splice Angle (rad)",fontsize=18)\nplt.ylabel(r\'$\\theta_4$\'+ " Splice Angle (rad)",fontsize=18)' 
    }
    
    
    
    
    for k in range(0,l_iterations):    
        for l in range(0,l_iterations):
            for i in range(0,N):
                for j in range(0,N):
                    state_pbs = eval(sim_dict[sim_type])
                    
                    d_matrix = state_pbs * state_pbs.dag() # Alternatively one can use ket2dm().
                    
                    rho_pol = d_matrix.ptrace([0,2]) # Reduced density matrix with frequency traced over. 
                    concurrence_range[i][j] = concurrence(rho_pol)
            plt.figure()
            exec(theta_sweep_1d_plot[sim_type])
            exec(theta_sweep_1d_title[sim_type])
            plt.ylabel("Concurrence",fontsize=18)
            plt.xlabel("Splice Angle (rad)",fontsize=18)
            
            
            #This was just taken from StackOverflow. Using this to format x axis so splice angle in units of pi rad
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
            #End of copied section.
            
            
            plt.tick_params(axis='y', which='major', labelsize=18)
            plt.tick_params(axis='x', which='major', labelsize=20)
            ax.set_ylim([0,1])
            plt.figure()
            
            
            #Same as above. Copied from stack overflow to format x and y. 
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
            ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
            #End of copied section 
            
            
            cmap = eval(theta_sweep_2d_plot[sim_type])
            plt.clim(0,1) #Fixed colour scale.
            plt.colorbar(cmap)
            
            #These dictionaries contain code that would be executed to format the plot and axis titles. 
            exec(theta_sweep_2d_title[sim_type])
            exec(theta_sweep_2d_axes[sim_type])
            plt.tick_params(axis='y', which='major', labelsize=20)
            plt.tick_params(axis='x', which='major', labelsize=20)
            concurrence_range = np.zeros([N,N])
    plt.show()
    return

def get_results_hom_dip():
    pass

def get_histogram_4x():
    n_bins = 30
    n_samples = 1000
    
    theta_dist = get_truncated_normal(np.pi/2, np.pi/36, 0, np.pi)
    theta_1 = theta_dist.rvs(size=n_samples)
    theta_2 = theta_dist.rvs(size=n_samples)
    theta_3 = theta_dist.rvs(size=n_samples)
    theta_4 = theta_dist.rvs(size=n_samples)
    
    length_dist = get_truncated_normal(0.965, 0.01, 0.95, 0.98)
    l_1 = length_dist.rvs(size=n_samples)
    l_1p = length_dist.rvs(size=n_samples)
    l_2 = length_dist.rvs(size=n_samples)
    l_2p = length_dist.rvs(size=n_samples)
    l_3 = length_dist.rvs(size=n_samples)
    l_3p = length_dist.rvs(size=n_samples)
    l_4 = length_dist.rvs(size=n_samples)
    l_4p = length_dist.rvs(size=n_samples)
    
    concurrence_range = np.zeros([n_samples])
    
    for i in range(0,n_samples-1):
        state_out = define_state_4x(theta_1[i],theta_2[i],theta_3[i], theta_4[i], l_1[i],l_1p[i],l_2[i],l_2p[i],l_3[i],l_3p[i],l_4[i],l_4p[i])
        d_matrix = state_out * state_out.dag()
        #alternatively can use ket2dm
        rho_pol = d_matrix.ptrace([0,2])
        concurrence_range[i] = concurrence(rho_pol)
        
    fig, ax = plt.subplots(1, 1)
    ax.hist(concurrence_range, density=False, histtype='stepfilled', alpha=0.2, bins=n_bins)
    ax.legend(loc='best', frameon=False)
    ax.set_ylim([0,1000])
    plt.title(r'Recorded Concurrence Values for Four Splice Configuration')
    plt.ylabel("Number of Occurences")
    plt.xlabel("Concurrence")
    plt.show()
    return

def get_histogram_2x():
    n_bins = 30
    n_samples = 1000
    
    theta_dist = get_truncated_normal(np.pi/2, np.pi/36, 0, np.pi)
    theta_1 = theta_dist.rvs(size=n_samples)
    theta_2 = theta_dist.rvs(size=n_samples)
    
    length_dist = get_truncated_normal(0.965, 0.01, 0.95, 0.98)
    l_1 = length_dist.rvs(size=n_samples)
    l_1p = length_dist.rvs(size=n_samples)
    l_2 = length_dist.rvs(size=n_samples)
    l_2p = length_dist.rvs(size=n_samples)

    
    concurrence_range = np.zeros([n_samples])
    
    for i in range(0,n_samples-1):
        state_out = define_state(theta_1[i],theta_2[i],l_1[i],l_1p[i],l_2[i],l_2p[i])
        d_matrix = state_out * state_out.dag()
        #alternatively can use ket2dm
        rho_pol = d_matrix.ptrace([0,2])
        concurrence_range[i] = concurrence(rho_pol)
        
    fig, ax = plt.subplots(1, 1)
    ax.hist(concurrence_range, density=False, histtype='stepfilled', alpha=0.2,bins=n_bins)
    ax.legend(loc='best', frameon=False)
    cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
    ax.set_ylim([0,1000])
    plt.title(r'Recorded Concurrence Values for Two Splice Configuration')
    plt.ylabel("Number of Occurences")
    plt.xlabel("Concurrence")
    plt.show()
    return

#This was just coppied from github. See more information on truncnorm documentation.
def get_truncated_normal(mean, sd, low, upp):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#WARNING: THE CODE BELOW WAS COPIED FROM STACKOVERFLOW.
#Use this to get scale in terms of pi...
def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))



single_settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.000,
        'l2':           1.000,
        'l2_p':         1.000,                                  # lengths have similar effect as Calvin's thesis
        'a1':           80.0 * np.pi/180.0,                      # !!! angle has less effect than in Calvin's thesis
        'a2':           80.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),                            # power imbalance has similar effect to Calvin's thesis
        #'filter_range': np.linspace(0,0.5*10e12,N)*2*np.pi      # assume max PPSF bandwidth of 10THz?
    }   
state_pbs = define_state(single_settings['a1'],single_settings['a2'],single_settings['l1'],single_settings['l1_p'],single_settings['l2'],single_settings['l2_p'])
d_matrix = d_matrix = ket2dm(state_pbs)
rho_pol = d_matrix.ptrace([0,2])
print(concurrence(rho_pol))

# get_histogram_2x()
# get_histogram_4x()
# get_results_2x()
#l_mismatch_2x()
##theta_l_mismatch_2x()
# get_results_4x()
# state_out = define_state_4x(np.pi/2,np.pi/2,np.pi/2, np.pi/2, 0.9,0.95,0.9,0.95,0.9,1,0.9,0.9)
# rho_pol = state_out.ptrace([0,2])
# conc = concurrence(rho_pol)
# print(conc)
