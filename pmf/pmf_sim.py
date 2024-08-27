import numpy as np

def oz_pmf_pigtail_int(alpha, beta, gamma, L1, L2, L3, input_state, wvl, B):
    crot = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
        ])
    xrot = np.array([
        [np.cos(beta), -np.sin(beta)],
        [np.sin(beta), np.cos(beta)]
        ])
    out_rot = np.array([
        [np.cos(gamma), -np.sin(gamma)],
        [np.sin(gamma), np.cos(gamma)]
        ])
    norm_in = input_state/np.linalg.norm(input_state) # normalize the input state
    phi1 = np.pi*B*L1 / wvl
    phi2 = np.pi*B*L2 / wvl
    phi3 = np.pi*B*L3 / wvl

    J_pmf1 = np.array([
        [np.exp(1j*phi1),0],
        [0,np.exp(-1j*phi1)]
    ])
    J_pmf2 = np.array([
        [np.exp(1j*phi2),0],
        [0,np.exp(-1j*phi2)]
    ])
    J_pmf3 = np.array([
        [np.exp(1j*phi3),0],
        [0,np.exp(-1j*phi3)]
    ])
    state_out = np.dot(out_rot,np.dot(J_pmf3,np.dot(xrot,np.dot(J_pmf2,np.dot(crot,np.dot(J_pmf1,norm_in))))))
    
    return state_out