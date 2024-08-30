import numpy as np
import matplotlib.pyplot as plt
from qutip import concurrence

from heps_state import *

def plot_length_angle_mismatch(title, matrix1=[], mx1_xlabel=[], mx1_xticks=[], mx1_ylabel=[], mx1_yticks=[], matrix2=[], mx2_xlabel=[], mx2_xticks=[], mx2_ylabel=[], mx2_yticks=[]):
    # Create new figure for this test
    if matrix1 == [] or matrix2 == []:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        ax2 = ax1
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(title, fontsize=24)

    if not matrix1 == []:
        im1 = ax1.imshow(matrix1, cmap='viridis', interpolation='nearest')
        # Generate tick positions and labels
        tick_positions = np.linspace(0, matrix1.shape[1] - 1, num=5)  # 5 ticks for simplicity
        # xtick_labels = np.linspace(min(mm), max(mm), num=5)  
        # ytick_labels = np.linspace(min(dws), max(dws), num=5)
        ax1.set_xticks(tick_positions, labels=np.round(mx1_xticks, 3))
        ax1.set_yticks(tick_positions, labels=np.round(mx1_yticks, 3))
        ax1.invert_yaxis()
        fig.colorbar(im1, ax=ax1, label='Concurrence')
        ax1.set_xlabel(mx1_xlabel)
        ax1.set_ylabel(mx1_ylabel)
        ax1.set_title('Effect of Length Mismatch')

    if not matrix2 == []:
        im2 = ax2.imshow(matrix2, cmap='viridis', interpolation='nearest')
        tick_positions = np.linspace(0, matrix2.shape[1] - 1, num=5)
        # xtick_labels = np.linspace(min(tt), max(tt), num=5)  
        # ytick_labels = np.linspace(min(dws), max(dws), num=5)
        ax2.set_xticks(tick_positions, labels=np.round(mx2_xticks, 3))
        ax2.set_yticks(tick_positions, labels=np.round(mx2_yticks, 3))
        ax2.invert_yaxis()
        fig.colorbar(im2, ax=ax2, label='Concurrence')
        ax2.set_xlabel(mx2_xlabel)
        ax2.set_ylabel(mx2_ylabel)
        ax2.set_title('Effect of Angular Mismatch')

    return fig, ax1, ax2

def conc_length_mismatch(settings, m1, m2):
    settings['l1_p'] = settings['l1'] + m1
    settings['l2_p'] = settings['l2'] + m2
    he_l = []
    for i in range(0,len(settings['filter_range'])):
        he_state = define_state(settings['filter_range'][i],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
        he_l.append(he_state)

    he_sum = sum(he_l)/len(settings['filter_range'])
    return concurrence(he_sum.ptrace([0,2]))
    
def conc_angle_mismatch(settings, t1, t2):
    settings['a1'] = (90+t1) * np.pi/180.0
    settings['a2'] = (90+t2) * np.pi/180.0
    he_l = []
    for i in range(0,len(settings['filter_range'])):
        he_state = define_state(settings['filter_range'][i],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
        he_l.append(he_state)

    he_sum = (sum(he_l))/len(settings['filter_range'])
    return concurrence(he_sum.ptrace([0,2]))

def single_setting_example(N=100):
    single_settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.020,
        'l2':           1.000,
        'l2_p':         0.980,                                  
        'a1':           90.0 * np.pi/180.0,                    
        'a2':           90.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),                 
        'filter_range': np.linspace(0.0e12,3.0e12,N)
    }   
    
    c_inft = concurrence(
        define_state(2.0e12, 
            single_settings['a1'],
            single_settings['a2'],
            single_settings['l1'],
            single_settings['l1_p'],
            single_settings['l2'],
            single_settings['l2_p']).ptrace([0,2])
    )

    he_l = []
    for i in range(0,N):
        he_state = define_state(single_settings['filter_range'][i],single_settings['a1'],single_settings['a2'],single_settings['l1'],single_settings['l1_p'],single_settings['l2'],single_settings['l2_p'])#single_settings['p']
        he_l.append(he_state)
    
    he_sum = (sum(he_l))/N
    c_fint = concurrence(he_sum.ptrace([0,2]))
    return c_inft, c_fint

def length_angle_inft(dw, min_m, max_m, min_t, max_t, size=20):
    settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.000,
        'l2':           1.000,
        'l2_p':         1.000,                                  
        'a1':           90.0 * np.pi/180.0,                      
        'a2':           90.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),                            
        'filter_range': dw
    }

    def conc_length_mismatch(m1, m2):
        settings['l1_p'] = settings['l1'] + m1
        settings['l2_p'] = settings['l2'] + m2
    
        he_state = define_state(settings['filter_range'],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])

        return concurrence(he_state.ptrace([0,2]))
    
    def conc_angle_mismatch(t1, t2):
        settings['a1'] = (90+t1) * np.pi/180.0
        settings['a2'] = (90+t2) * np.pi/180.0
        
        he_state = define_state(settings['filter_range'],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
            
        return concurrence(he_state.ptrace([0,2]))

    # Generate the matrix using the concurrence function
    mm1 = np.linspace(min_m,max_m,size)
    mm2 = np.linspace(min_m,max_m,size)
    matrix1 = np.zeros((size, size))
    for j, m2 in enumerate(mm2):
        for i, m1 in enumerate(mm1):
            matrix1[j, i] = conc_length_mismatch(m1, m2)

    settings['l1_p'] = settings['l1']
    settings['l2_p'] = settings['l2']

    # Generate the matrix using the concurrence function
    tt1 = np.linspace(min_t,max_t,size)
    tt2 = np.linspace(min_t,max_t,size)
    matrix2 = np.zeros((size, size))
    for j, t2 in enumerate(tt2):
        for i, t1 in enumerate(tt1):
            matrix2[j, i] = conc_angle_mismatch(t1, t2)

    # Create new figure for this test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)  # 1 row, 2 columns
    fig.suptitle(f' {dw/1e12}THz Detuning ', fontsize=24)
    im1 = ax1.imshow(matrix1, cmap='viridis', interpolation='nearest')
    # Generate tick positions and labels
    tick_positions = np.linspace(0, size - 1, num=5)  # 5 ticks for simplicity
    xtick_labels = np.linspace(min(mm1), max(mm1), num=5)  
    ytick_labels = np.linspace(min(mm2), max(mm2), num=5)
    ax1.set_xticks(tick_positions, labels=np.round(xtick_labels, 2))
    ax1.set_yticks(tick_positions, labels=np.round(ytick_labels, 2))
    ax1.invert_yaxis()
    fig.colorbar(im1, ax=ax1, label='Concurrence')
    ax1.set_xlabel('\\textit{$\\alpha_1$} (m)')
    ax1.set_ylabel('\\textit{$\\alpha_2$} (m)')
    ax1.set_title('Effect of Length Mismatch')

    im2 = ax2.imshow(matrix2, cmap='viridis', interpolation='nearest')
    tick_positions = np.linspace(0, size - 1, num=5)
    xtick_labels = np.linspace(min(tt1), max(tt1), num=5)  
    ytick_labels = np.linspace(min(tt2), max(tt2), num=5)
    ax2.set_xticks(tick_positions, labels=np.round(xtick_labels, 2))
    ax2.set_yticks(tick_positions, labels=np.round(ytick_labels, 2))
    ax2.invert_yaxis()
    fig.colorbar(im2, ax=ax2, label='Concurrence')
    ax2.set_xlabel('\\textit{$t_1$} (deg)')
    ax2.set_ylabel('\\textit{$t_2$} (deg)')
    ax2.set_title('Effect of Angular Mismatch')

    return fig, ax1, ax2

def length_angle_inft_slices(dws, min_m, max_m, min_t, max_t, size=20):
    settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.000,
        'l2':           1.000,
        'l2_p':         1.000,                                  
        'a1':           90.0 * np.pi/180.0,                      
        'a2':           90.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),                            
        'filter_range': dws[0]
    }

    def conc_length_mismatch(dw, m1, m2):
        settings['l1_p'] = settings['l1'] + m1
        settings['l2_p'] = settings['l2'] + m2
    
        he_state = define_state(dw,settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])

        return concurrence(he_state.ptrace([0,2]))
    
    def conc_angle_mismatch(dw, t1, t2):
        settings['a1'] = (90+t1) * np.pi/180.0
        settings['a2'] = (90+t2) * np.pi/180.0
        
        he_state = define_state(dw,settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
            
        return concurrence(he_state.ptrace([0,2]))

    # Generate the matrix using the concurrence function
    mm = np.linspace(min_m,max_m,size)
    matrix1 = np.zeros((len(dws), size))
    for j, m in enumerate(mm):
        for i, dw in enumerate(dws):
            matrix1[i, j] = conc_length_mismatch(dw, m, m)

    settings['l1_p'] = settings['l1']
    settings['l2_p'] = settings['l2']

    # Generate the matrix using the concurrence function
    tt = np.linspace(min_t,max_t,size)
    matrix2 = np.zeros((len(dws), size))
    for j, t in enumerate(tt):
        for i, dw in enumerate(dws):
            matrix2[i, j] = conc_angle_mismatch(dw, t, t)
    

    print(np.min(matrix2), np.max(matrix2))

    dws = dws / 1e12
    xtick_labels1 = np.linspace(min(mm), max(mm), num=5)  
    ytick_labels1 = np.linspace(min(dws), max(dws), num=5)
    xlabel1 = '\\textit{$\\alpha$} (m)'
    ylabel1 = '\\textit{$d\\omega$} (THz)'
    xtick_labels2 = np.linspace(min(tt), max(tt), num=5)  
    ytick_labels2 = np.linspace(min(dws), max(dws), num=5)
    xlabel2 = '\\textit{$t$} (deg)'
    ylabel2 = '\\textit{$d\\omega$} (THz)'

    return plot_length_angle_mismatch(' Infinitesimal Bin Detuning ',
                               matrix1=matrix1, 
                               mx1_xlabel=xlabel1, 
                               mx1_xticks=xtick_labels1, 
                               mx1_ylabel=ylabel1,
                               mx1_yticks=ytick_labels1,
                               matrix2=matrix2,
                               mx2_xlabel=xlabel2, 
                               mx2_xticks=xtick_labels2, 
                               mx2_ylabel=ylabel2,
                               mx2_yticks=ytick_labels2
                               )

def length_angle_st(bw, min_m, max_m, min_t, max_t, N=100, size=20):
    settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.000,
        'l2':           1.000,
        'l2_p':         1.000,                                  
        'a1':           90.0 * np.pi/180.0,                      
        'a2':           90.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),                            
        'filter_range': np.linspace(0.0e12,0.5*bw,N)      
    }

    def conc_length_mismatch(m1, m2):
        settings['l1_p'] = settings['l1'] + m1
        settings['l2_p'] = settings['l2'] + m2
        he_l = []
        for i in range(0,N):
            he_state = define_state(settings['filter_range'][i],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
            he_l.append(he_state)
    
        he_sum = (sum(he_l))/N
        return concurrence(he_sum.ptrace([0,2]))
    
    def conc_angle_mismatch(t1, t2):
        settings['a1'] = (90+t1) * np.pi/180.0
        settings['a2'] = (90+t2) * np.pi/180.0
        he_l = []
        for i in range(0,N):
            he_state = define_state(settings['filter_range'][i],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
            he_l.append(he_state)
    
        he_sum = (sum(he_l))/N
        return concurrence(he_sum.ptrace([0,2]))

    # Generate the matrix using the concurrence function
    mm1 = np.linspace(min_m,max_m,size)
    mm2 = np.linspace(min_m,max_m,size)
    matrix1 = np.zeros((size, size))
    for j, m2 in enumerate(mm2):
        for i, m1 in enumerate(mm1):
            matrix1[j, i] = conc_length_mismatch(m1, m2)

    settings['l1_p'] = settings['l1']
    settings['l2_p'] = settings['l2']

    # Generate the matrix using the concurrence function
    tt1 = np.linspace(min_t,max_t,size)
    tt2 = np.linspace(min_t,max_t,size)
    matrix2 = np.zeros((size, size))
    for j, t2 in enumerate(tt2):
        for i, t1 in enumerate(tt1):
            matrix2[j, i] = conc_angle_mismatch(t1, t2)

    # Create new figure for this test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)  # 1 row, 2 columns
    fig.suptitle(f' {bw/1e12}THz Bandwidth ', fontsize=24)
    im1 = ax1.imshow(matrix1, cmap='viridis', interpolation='nearest')
    # Generate tick positions and labels
    tick_positions = np.linspace(0, size - 1, num=5)  # 5 ticks for simplicity
    xtick_labels = np.linspace(min(mm1), max(mm1), num=5)  
    ytick_labels = np.linspace(min(mm2), max(mm2), num=5)
    ax1.set_xticks(tick_positions, labels=np.round(xtick_labels, 2))
    ax1.set_yticks(tick_positions, labels=np.round(ytick_labels, 2))
    ax1.invert_yaxis()
    fig.colorbar(im1, ax=ax1, label='Concurrence')
    ax1.set_xlabel('\\textit{$\\alpha_1$} (m)')
    ax1.set_ylabel('\\textit{$\\alpha_2$} (m)')
    ax1.set_title('Effect of Length Mismatch')

    im2 = ax2.imshow(matrix2, cmap='viridis', interpolation='nearest')
    tick_positions = np.linspace(0, size - 1, num=5)
    xtick_labels = np.linspace(min(tt1), max(tt1), num=5)  
    ytick_labels = np.linspace(min(tt2), max(tt2), num=5)
    ax2.set_xticks(tick_positions, labels=np.round(xtick_labels, 2))
    ax2.set_yticks(tick_positions, labels=np.round(ytick_labels, 2))
    ax2.invert_yaxis()
    fig.colorbar(im2, ax=ax2, label='Concurrence')
    ax2.set_xlabel('\\textit{$t_1$} (deg)')
    ax2.set_ylabel('\\textit{$t_2$} (deg)')
    ax2.set_title('Effect of Angular Mismatch')

def length_angle_st_slices(dws, min_m, max_m, min_t, max_t, N=100, size=20):
    settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.000,
        'l2':           1.000,
        'l2_p':         1.000,                                  
        'a1':           90.0 * np.pi/180.0,                      
        'a2':           90.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),                            
        'filter_range': dws[0]
    }

    def conc_length_mismatch(bw, m1, m2):
        settings['l1_p'] = settings['l1'] + m1
        settings['l2_p'] = settings['l2'] + m2
        dd = np.linspace(0.0e12,0.5*bw,N)
        he_l = []
        for i in range(0,N):
            he_state = define_state(dd[i],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
            he_l.append(he_state)
    
        he_sum = (sum(he_l))/N
        return concurrence(he_sum.ptrace([0,2]))
    
    def conc_angle_mismatch(bw, t1, t2):
        settings['a1'] = (90+t1) * np.pi/180.0
        settings['a2'] = (90+t2) * np.pi/180.0
        dd = np.linspace(0.0e12,0.5*bw,N)
        he_l = []
        for i in range(0,N):
            he_state = define_state(dd[i],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
            he_l.append(he_state)

        he_sum = (sum(he_l))/N
        return concurrence(he_sum.ptrace([0,2]))

    # Generate the matrix using the concurrence function
    mm = np.linspace(min_m,max_m,size)
    matrix1 = np.zeros((len(dws), size))
    for j, m in enumerate(mm):
        for i, dw in enumerate(dws):
            matrix1[i, j] = conc_length_mismatch(dw, m, m)

    settings['l1_p'] = settings['l1']
    settings['l2_p'] = settings['l2']

    # Generate the matrix using the concurrence function
    tt = np.linspace(min_t,max_t,size)
    matrix2 = np.zeros((len(dws), size))
    for j, t in enumerate(tt):
        for i, dw in enumerate(dws):
            matrix2[i, j] = conc_angle_mismatch(dw, t, t)
    

    print(np.min(matrix2), np.max(matrix2))

    dws = dws / 1e12
    xtick_labels1 = np.linspace(min(mm), max(mm), num=5)  
    ytick_labels1 = np.linspace(min(dws), max(dws), num=5)
    xlabel1 = '\\textit{$\\alpha$} (m)'
    ylabel1 = '\\textit{$bw$} (THz)'
    xtick_labels2 = np.linspace(min(tt), max(tt), num=5)  
    ytick_labels2 = np.linspace(min(dws), max(dws), num=5)
    xlabel2 = '\\textit{$t$} (deg)'
    ylabel2 = '\\textit{$bw$} (THz)'

    plot_length_angle_mismatch(' Varying Top-hat Bandwidth at Degeneracy ',
                               matrix1=matrix1, 
                               mx1_xlabel=xlabel1, 
                               mx1_xticks=xtick_labels1, 
                               mx1_ylabel=ylabel1,
                               mx1_yticks=ytick_labels1,
                               matrix2=matrix2,
                               mx2_xlabel=xlabel2, 
                               mx2_xticks=xtick_labels2, 
                               mx2_ylabel=ylabel2,
                               mx2_yticks=ytick_labels2
                               )

def length_angle_dt(bw, dw, min_m, max_m, min_t, max_t, N=100, size=20):
    settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.000,
        'l2':           1.000,
        'l2_p':         1.000,                                  
        'a1':           90.0 * np.pi/180.0,                      
        'a2':           90.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),                            
        'filter_range': np.linspace(dw-0.5*bw,dw+0.5*bw,N)      
    }

    

    # Generate the matrix using the concurrence function
    mm1 = np.linspace(min_m,max_m,size)
    mm2 = np.linspace(min_m,max_m,size)
    matrix1 = np.zeros((size, size))
    for j, m2 in enumerate(mm2):
        for i, m1 in enumerate(mm1):
            matrix1[j, i] = conc_length_mismatch(m1, m2)

    settings['l1_p'] = settings['l1']
    settings['l2_p'] = settings['l2']

    # Generate the matrix using the concurrence function
    tt1 = np.linspace(min_t,max_t,size)
    tt2 = np.linspace(min_t,max_t,size)
    matrix2 = np.zeros((size, size))
    for j, t2 in enumerate(tt2):
        for i, t1 in enumerate(tt1):
            matrix2[j, i] = conc_angle_mismatch(t1, t2)

    # Create new figure for this test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)  # 1 row, 2 columns
    fig.suptitle(f'{dw/1e12}THz Detuned, {bw/1e12}THz Bandwidth ', fontsize=24)
    im1 = ax1.imshow(matrix1, cmap='viridis', interpolation='nearest')
    # Generate tick positions and labels
    tick_positions = np.linspace(0, size - 1, num=5)  # 5 ticks for simplicity
    xtick_labels = np.linspace(min(mm1), max(mm1), num=5)  
    ytick_labels = np.linspace(min(mm2), max(mm2), num=5)
    ax1.set_xticks(tick_positions, labels=np.round(xtick_labels, 2))
    ax1.set_yticks(tick_positions, labels=np.round(ytick_labels, 2))
    ax1.invert_yaxis()
    fig.colorbar(im1, ax=ax1, label='Concurrence')
    ax1.set_xlabel('\\textit{$\\alpha_1$} (m)')
    ax1.set_ylabel('\\textit{$\\alpha_2$} (m)')
    ax1.set_title('Effect of Length Mismatch')

    im2 = ax2.imshow(matrix2, cmap='viridis', interpolation='nearest')
    tick_positions = np.linspace(0, size - 1, num=5)
    xtick_labels = np.linspace(min(tt1), max(tt1), num=5)  
    ytick_labels = np.linspace(min(tt2), max(tt2), num=5)
    ax2.set_xticks(tick_positions, labels=np.round(xtick_labels, 2))
    ax2.set_yticks(tick_positions, labels=np.round(ytick_labels, 2))
    ax2.invert_yaxis()
    fig.colorbar(im2, ax=ax2, label='Concurrence')
    ax2.set_xlabel('\\textit{$t_1$} (deg)')
    ax2.set_ylabel('\\textit{$t_2$} (deg)')
    ax2.set_title('Effect of Angular Mismatch')

def length_angle_dt_slices(bw, dws, min_m, max_m, min_t, max_t, N=100, size=20):
    settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.000,
        'l2':           1.000,
        'l2_p':         1.000,                                  
        'a1':           90.0 * np.pi/180.0,                      
        'a2':           90.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),                            
        #'filter_range': np.linspace(dw-0.5*bw,dw+0.5*bw,N)      
    }

    def conc_length_mismatch(dw, m1, m2):
        settings['l1_p'] = settings['l1'] + m1
        settings['l2_p'] = settings['l2'] + m2
        he_l = []
        dd=np.linspace(dw-0.5*bw,dw+0.5*bw,N)
        for i in range(0,N):
            he_state = define_state(dd[i],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
            he_l.append(he_state)
    
        he_sum = (sum(he_l))/N
        return concurrence(he_sum.ptrace([0,2]))
    
    def conc_angle_mismatch(dw, t1, t2):
        settings['a1'] = (90+t1) * np.pi/180.0
        settings['a2'] = (90+t2) * np.pi/180.0
        he_l = []
        dd=np.linspace(dw-0.5*bw,dw+0.5*bw,N)
        for i in range(0,N):
            he_state = define_state(dd[i],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
            he_l.append(he_state)
    
        he_sum = (sum(he_l))/N
        return concurrence(he_sum.ptrace([0,2]))

    # Generate the matrix using the concurrence function
    mm = np.linspace(min_m,max_m,size)
    matrix1 = np.zeros((len(dws), size))
    for j, m in enumerate(mm):
        for i, dw in enumerate(dws):
            matrix1[i, j] = conc_length_mismatch(dw, m, m)

    settings['l1_p'] = settings['l1']
    settings['l2_p'] = settings['l2']

    # Generate the matrix using the concurrence function
    tt = np.linspace(min_t,max_t,size)
    matrix2 = np.zeros((len(dws), size))
    for j, t in enumerate(tt):
        for i, dw in enumerate(dws):
            matrix2[i, j] = conc_angle_mismatch(dw, t, t)
    

    print(np.min(matrix2), np.max(matrix2))

    dws = dws / 1e12
    xtick_labels1 = np.linspace(min(mm), max(mm), num=5)  
    ytick_labels1 = np.linspace(min(dws), max(dws), num=5)
    xlabel1 = '\\textit{$\\alpha$} (m)'
    ylabel1 = '\\textit{$d\\omega$} (THz)'
    xtick_labels2 = np.linspace(min(tt), max(tt), num=5)  
    ytick_labels2 = np.linspace(min(dws), max(dws), num=5)
    xlabel2 = '\\textit{$t$} (deg)'
    ylabel2 = '\\textit{$d\\omega$} (THz)'

    plot_length_angle_mismatch(f' Varying Detuning of {bw}THz Dual-tooth Filter',
                               matrix1=matrix1, 
                               mx1_xlabel=xlabel1, 
                               mx1_xticks=xtick_labels1, 
                               mx1_ylabel=ylabel1,
                               mx1_yticks=ytick_labels1,
                               matrix2=matrix2,
                               mx2_xlabel=xlabel2, 
                               mx2_xticks=xtick_labels2, 
                               mx2_ylabel=ylabel2,
                               mx2_yticks=ytick_labels2
                               )

def powerimbalan_st(bw, min_p, max_p, N=100, size=20):
    settings = {
        'lambda_deg':   1556e-9,
        'l1':           1.000,
        'l1_p':         1.000,
        'l2':           1.000,
        'l2_p':         1.000,                                  
        'a1':           90.0 * np.pi/180.0,                      
        'a2':           90.0 * np.pi/180.0,
        'p':            np.sqrt(0.5),                            
        'filter_range': np.linspace(0.0e12,0.5*bw,N)
    }

    # Generate the array using the concurrence function
    pp = np.linspace(min_p,max_p,size)
    
    array = np.zeros(size)
    for j, p in enumerate(pp):
        he_l = []
        for i in range(0,N):
            he_state = define_state(settings['filter_range'][i],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'],p)
            he_l.append(he_state)
    
        he_sum = (sum(he_l))/N
        array[j] = concurrence(he_sum.ptrace([0,2]))

    # Create new figure for this test
    fig, ax1 = plt.subplots(figsize=(6, 6))
    #ax1 = fig.add_subplot(111)
    title = ax1.set_title(f'{bw/1e12}THz Bandwidth - Effect of Power Imbalance', fontsize=16, pad=12)
    title.set_y(1.3)
    fig.subplots_adjust(top=0.85)
    ax1.plot(pp/np.sqrt(0.5), array)
    ax1.grid(True)
    ax1.set_ylabel('Concurrence')
    ax1.set_xlabel('\\textit{p ($\sqrt{1/2}$)}')

    ax2 = ax1.twiny()
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([np.round((x*np.sqrt(0.5))**2, 2) for x in ax1.get_xticks()])
    ax2.set_xlabel('$R_{ps}$')
