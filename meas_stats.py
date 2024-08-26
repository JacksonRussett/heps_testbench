import numpy as np
import QuantumTomography as qKLib
from qutip import *
from qutip.measurement import measure_observable, measurement_statistics

def measure_pc_counts(dm, meas_state):
    if dm.isket:
        dm = ket2dm(dm)
    if meas_state.isket:
        meas_state = ket2dm(meas_state)

    # Simple projective measurement statistics
    a,collapsed_states,probs = measurement_statistics(dm.unit(), meas_state.unit())

    # Do the following to simulate action of a single measurements
    results = {0.0: 0, 1.0: 0}
    for _ in range(1000):
        value, new_state = measure_observable(dm.unit(), meas_state.unit())
        results[round(value)] += 1

    return probs, results

def brute_counts(dm, meas_state):
    _, results = measure_pc_counts(dm, meas_state)
    return results[1.0]

def quick_counts(dm, meas_state, max_cc, do_rnd=True):
    if dm.isket:
        dm = ket2dm(dm)
    if meas_state.isket:
        meas_state = ket2dm(meas_state)

    # Simple projective measurement statistics
    _,_,probs = measurement_statistics(dm.unit(), meas_state.unit())
    #print(probs)
    if do_rnd:
        return round(np.random.poisson(np.real(probs[-1])*max_cc,1)[0])
    else: 
        return probs[-1]*max_cc

def gen_tomo_input(dm, max_cc, add_noise=True):
    H = basis(2, 0)
    V = basis(2, 1)
    D = (H + V).unit()
    A = (H - V).unit()
    R = (H + 1j*V).unit()
    L = (H - 1j*V).unit()

    tomo_input = np.array([
        [1,0,0,quick_counts(dm, tensor(H,H), max_cc, add_noise),   1,0,1,0],
        [1,0,0,quick_counts(dm, tensor(H,V), max_cc, add_noise),   1,0,0,1],
        [1,0,0,quick_counts(dm, tensor(H,D), max_cc, add_noise),   1,0,0.7071,0.7071],
        [1,0,0,quick_counts(dm, tensor(H,R), max_cc, add_noise),   1,0,0.7071,0.7071j],
        [1,0,0,quick_counts(dm, tensor(V,H), max_cc, add_noise),   0,1,1,0],
        [1,0,0,quick_counts(dm, tensor(V,V), max_cc, add_noise),   0,1,0,1],
        [1,0,0,quick_counts(dm, tensor(V,D), max_cc, add_noise),   0,1,0.7071,0.7071],
        [1,0,0,quick_counts(dm, tensor(V,R), max_cc, add_noise),   0,1,0.7071,0.7071j],
        [1,0,0,quick_counts(dm, tensor(D,H), max_cc, add_noise),   0.7071,0.7071,1,0],
        [1,0,0,quick_counts(dm, tensor(D,V), max_cc, add_noise),   0.7071,0.7071,0,1],
        [1,0,0,quick_counts(dm, tensor(D,D), max_cc, add_noise),   0.7071,0.7071,0.7071,0.7071],
        [1,0,0,quick_counts(dm, tensor(D,R), max_cc, add_noise),   0.7071,0.7071,0.7071,0.7071j],
        [1,0,0,quick_counts(dm, tensor(R,H), max_cc, add_noise),   0.7071,0.7071j,1,0],
        [1,0,0,quick_counts(dm, tensor(R,V), max_cc, add_noise),   0.7071,0.7071j,0,1],
        [1,0,0,quick_counts(dm, tensor(R,D), max_cc, add_noise),   0.7071,0.7071j,0.7071,0.7071],
        [1,0,0,quick_counts(dm, tensor(R,R), max_cc, add_noise),   0.7071,0.7071j,0.7071,0.7071j]
    ])

    return tomo_input