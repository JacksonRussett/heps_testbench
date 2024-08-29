import numpy as np
import matplotlib.pyplot as plt
from qutip import concurrence, basis, tensor
from pmf.pmf_sim import *
from heps_state import *
from meas_stats import quick_counts, gen_tomo_input

def single_setting_example(tomo, N=100):
     ## Static Params
    N = 100                                 # number of slices in finite bandwidth approx
    src_brightness = 1000                # baseline brightness, Hz
    bw = 6.0e12                             # bandwdith of single flat-top filter, Hz
    src_settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.020,
        'l2':           1.000,
        'l2_p':         1.030,                                  
        'a1':           88.0 * np.pi/180.0,                      
        'a2':           87.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),       # this will be ignored for this simulation                          
        'filter_range': np.linspace(0.0,0.5*bw,N)   # this range defines the flat-top filter
    }
    
    ## Dynamic Params
    num_meas = 16                           # number of tomography measurments - this shouldnt be changed!
    Tacq = 10.0                             # acquisition time of each tomo measurement 
    add_noise = False                        # add poisson noise to simulated counts
    # define the power split / brightness fluctuation params
    fluc_settings = {
        'mode'   : 'direct',                  # 'direct' - direct sinusoid of split over time, 'temp' - use simulation of PMF with sinusoid of temp with time   
        'period' : 197,
        'amplit' : 0.3,                     # amplitude of sinusoidal oscillation
        'baseln' : 0.5,                     # average/baseline of oscillation
        'phisft' : np.pi / 12.0,
        'tresol' : 1.0,                     # resolution of the time series, s
        'alpha'  : 0.0 * np.pi / 180.0,
        'beta'   : 0.0 * np.pi / 180.0,
        'gamma'  : 45. * np.pi / 180.0,
        'L1'     : 3.00,
        'L2'     : 1.00,
        'L3'     : 1.00,
        'lstate' : np.array([[0.0],[1]]),
        'lwvl'   : 778e-9,
        'B0'     : 5e-4,
        'DBt'    : -5.6e-7
    }

    ## Other params
    do_time_plot = False

    (rho, intens, fval) = dynamic_pwrimb_sim(tomo, src_settings, fluc_settings, num_meas, Tacq, src_brightness, add_noise, do_time_plot)
    print('C =\t', concurrence(rho))
    print('I_est =\t', intens)
    print('fval =\t', fval)

    # Display the plots
    if do_time_plot:
        plt.show() 

def direct_pwrimb_time_series(fluc_settings, num_periods=2):
    T=fluc_settings['period']
    A=fluc_settings['amplit'] 
    C=fluc_settings['baseln'] 
    phi=fluc_settings['phisft']
    res=fluc_settings['tresol']
    t = np.linspace(0, T*num_periods, int(T*num_periods/res))
    R_ps = A*np.sin(2*np.pi*t/T + phi)+C
    p = np.sqrt(R_ps)
    return (t, R_ps, p)

def pmf_pwrimb_time_series(fluc_settings, num_periods=2):
    T=fluc_settings['period']
    A=fluc_settings['amplit'] 
    C=fluc_settings['baseln'] 
    phi=fluc_settings['phisft']
    res=fluc_settings['tresol']
    t = np.linspace(0, T*num_periods, int(T*num_periods/res))
    T_pmf = A*np.sin(2*np.pi*t/T + phi)+C
    # power split ratio, R_ps = I1 / (I1 + I2)
    delta_b = T_pmf*fluc_settings['DBt']
    B = fluc_settings['B0'] + delta_b
    polh = np.array([
            [1, 0],
            [0, 0]
        ])
    R_ps = np.zeros(shape(T_pmf))
    for j in range(len(R_ps)):
        out = oz_pmf_pigtail_int(fluc_settings['alpha'], 
                                 fluc_settings['beta'], 
                                 fluc_settings['gamma'], 
                                 fluc_settings['L1'], 
                                 fluc_settings['L2'], 
                                 fluc_settings['L3'], 
                                 fluc_settings['lstate'], 
                                 fluc_settings['lwvl'], 
                                 B[j]
                                 )[:,0]
        R_ps[j] = np.dot(np.dot(polh,out).conj().T, np.dot(polh,out))
    p = np.sqrt(R_ps)
    return (t, R_ps, p)

def dynamic_pwrimb_sim(tomo, static_settings, fluc_settings, num_meas, Tacq, src_brightness, add_noise=False, do_time_plot=False):
    max_cc = Tacq * src_brightness
    if fluc_settings['mode'] == 'direct':
        (t, R_ps, p) = direct_pwrimb_time_series(fluc_settings)
    else:
        (t, R_ps, p) = pmf_pwrimb_time_series(fluc_settings)
    k_int = 1.0 - 2*np.abs(0.5 - R_ps)      # intensity/brightness multiplier

    if do_time_plot:
        fig, ax1 = plt.subplots(figsize=(6, 6))
        ax1.plot(t, R_ps)
        ax1.plot(t, p)
        ax1.plot(t, k_int)
        ax1.grid(True)
        ax1.set_ylabel('magnitude')
        ax1.set_xlabel('time (s)')
        ax1.set_ylim(0.0, 1.0)
        ax1.legend(['$R_{ps}$','$p$', '$k_{int}$'])
        
        def plot_vertical_lines(ax, num_lines, x_range=(0, 10), color='b', linestyle='--', linewidth=1):
            # Calculate the x-coordinates for the vertical lines
            x_coords = np.linspace(x_range[0], x_range[1], num_lines)

            # Plot the vertical lines
            for x in x_coords:
                ax.axvline(x=x, color=color, linestyle=linestyle, linewidth=linewidth)

        # Plot vertical lines to represent the end of QST measurements
        plot_vertical_lines(ax1, num_lines=num_meas, x_range=(Tacq, num_meas*Tacq), color='r', linestyle='--', linewidth=0.8)
        ax1.set_xlim(0.0, (num_meas+1)*Tacq)

    # Generate tomography data
    H = basis(2, 0)
    V = basis(2, 1)
    D = (H + V).unit()
    A = (H - V).unit()
    R = (H + 1j*V).unit()
    L = (H - 1j*V).unit()

    tomo_input = np.array([
        [1,0,0, 0,   1,0,1,0],
        [1,0,0, 0,   1,0,0,1],
        [1,0,0, 0,   1,0,0.7071,0.7071],
        [1,0,0, 0,   1,0,0.7071,0.7071j],
        [1,0,0, 0,   0,1,1,0],
        [1,0,0, 0,   0,1,0,1],
        [1,0,0, 0,   0,1,0.7071,0.7071],
        [1,0,0, 0,   0,1,0.7071,0.7071j],
        [1,0,0, 0,   0.7071,0.7071,1,0],
        [1,0,0, 0,   0.7071,0.7071,0,1],
        [1,0,0, 0,   0.7071,0.7071,0.7071,0.7071],
        [1,0,0, 0,   0.7071,0.7071,0.7071,0.7071j],
        [1,0,0, 0,   0.7071,0.7071j,1,0],
        [1,0,0, 0,   0.7071,0.7071j,0,1],
        [1,0,0, 0,   0.7071,0.7071j,0.7071,0.7071],
        [1,0,0, 0,   0.7071,0.7071j,0.7071,0.7071j]
    ])
    qt_meas_state = [
        tensor(H,H),
        tensor(H,V),
        tensor(H,D),
        tensor(H,R),
        tensor(V,H),
        tensor(V,V),
        tensor(V,D),
        tensor(V,R),
        tensor(D,H),
        tensor(D,V),
        tensor(D,D),
        tensor(D,R),
        tensor(R,H),
        tensor(R,V),
        tensor(R,D),
        tensor(R,R)
    ]
    
    # simulate each tomography measurement
    N = len(static_settings['filter_range'])
    for j in range(0, num_meas):
        idx = int(((j+1)*Tacq/fluc_settings['tresol'])+1)
        print(p[idx], k_int[idx])
        he_l = []
        for i in range(0,N):
            he_state = define_state(static_settings['filter_range'][i],static_settings['a1'],static_settings['a2'],static_settings['l1'],static_settings['l1_p'],static_settings['l2'],static_settings['l2_p'],p[idx])
            he_l.append(he_state)
    
        he_sum = (sum(he_l))/N
        dm = he_sum.ptrace([0,2])
        exp_counts = quick_counts(dm, qt_meas_state[j], k_int[idx]*max_cc, add_noise)
        tomo_input[j,3] = np.round(exp_counts,0)

    # Perform state estimation
    # print(tomo_input)
    [rho, intens, fval] = tomo.StateTomography_Matrix(tomo_input)
    #qKLib.printLastOutput(tomo)
    return (Qobj(rho, dims=[[2,2],[2,2]]), intens, fval)

def mc_phase(tomo, num_trials=10, add_poisson_noise=False):
    ## Static Params
    N = 100                                 # number of slices in finite bandwidth approx
    src_brightness = 100                    # baseline brightness, Hz
    bw = 2.4e12                             # bandwdith of single flat-top filter, Hz
    static_settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.020,
        'l2':           1.000,
        'l2_p':         1.030,                                  
        'a1':           88.0 * np.pi/180.0,                      
        'a2':           87.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),       # this will be ignored for this simulation                          
        'filter_range': np.linspace(0.0,0.5*bw,N)   # this range defines the flat-top filter
    }

    ## Dynamic Params
    num_meas = 16                           # number of tomography measurments - this shouldnt be changed!
    Tacq = 10.0                             # acquisition time of each tomo measurement 
    # define the power split / brightness fluctuation params
    fluc_settings = {
        'period' : 400,
        'amplit' : 0.1,
        'baseln' : 0.5,
        'phisft' : np.pi/3,
        'tresol' : 1.0                      # resolution of the time series, s
    }

    concurrence_trials = []
    phi_rnd = np.random.uniform(0.0, 2*np.pi, size=num_trials)
    for i in range(0, num_trials):
        fluc_settings['phisft'] = phi_rnd[i]
        (rho, _, _) = dynamic_pwrimb_sim(tomo, static_settings, fluc_settings, num_meas, Tacq, src_brightness, add_noise=add_poisson_noise, do_time_plot=False)
        concurrence_trials.append(concurrence(rho))
        print(f'Finished trial {i+1} of {num_trials}...')
    return (concurrence_trials, phi_rnd, static_settings, fluc_settings)

def display_mc_singleparam_results(trial_data, mc_param, param_ax_label):
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot the first histogram
    axs[0].hist(trial_data, alpha=0.7, color='blue', edgecolor='black')
    axs[0].set_title('Histogram of Concurrence')
    axs[0].set_xlabel('Concurrence')
    axs[0].set_ylabel('Frequency')
    axs[0].set_xlim([0,1])
    axs[0].grid(True)

    # Plot the second histogram
    axs[1].hist(mc_param, alpha=0.7, color='green', edgecolor='black')
    axs[1].set_title(f'Histogram of {param_ax_label}')
    axs[1].set_xlabel(param_ax_label)
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True)

    # Adjust layout
    plt.tight_layout()

