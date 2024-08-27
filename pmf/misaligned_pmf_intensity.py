import numpy as np
import matplotlib.pyplot as plt
from pmf_sim import *

#np.set_printoptions(suppress=True, precision=4)
plt.rcParams['text.usetex'] =   True
plt.rcParams["font.family"] =   "serif"
plt.rcParams["font.size"] =     "14"

# polarizer (PBS - vertical port)
polv = np.array([
    [0, 0],
    [0, 1]
    ])

# polarizer (PBS - horizontal port)
polh = np.array([
    [1, 0],
    [0, 0]
    ])

wvl = 778e-9
L1 = 3.00e0            # length of PMF (meters)
L2 = 1.00e0            # length of PMF (meters)
L3 = 0.00e0            # length of PMF (meters)
## oz fiber reports ER of ~30 witch corresponds to ~2 deg misalignment
alpha = 2 / 180 * np.pi  # angular misalignment at connector (rad)
beta =  0 / 180 * np.pi  # angular misalignment at cross-splice (rad)
gamma = 1.0*np.pi/4      # angular of PMF output to polarizer (rad)
#input state
raw_in = np.array([[0.03],[1]]) # jones calc poln state
n = 301               # number of points in sweep
delta_t = np.linspace(-2, 2, n)                    
delta_b = delta_t*-5.6e-7 # birefringence sweep -2C-+2C
B = 5e-4 + delta_b      # birefringence of PMF (a.u.)
print(B)

int_out1 = np.zeros((n,1))
int_out2 = np.zeros((n,1))
out = np.zeros((2, n), np.complex128)
for j in range(n):
    out[:,j] = oz_pmf_pigtail_int(alpha, beta, gamma, L1, L2, L3, raw_in, wvl, B[j])[:,0]
    int_out1[j] = np.dot(np.dot(polh,out[:,j]).conj().T, np.dot(polh,out[:,j]))
    int_out2[j] = np.dot(np.dot(polv,out[:,j]).conj().T, np.dot(polv,out[:,j]))

delta_int = max(int_out1)-min(int_out1)
print(delta_int)
print(np.sqrt(0.5*delta_int))

fig, ax1 = plt.subplots()
ax2 = ax1.twiny()

# Plot data on the birefringence x-axis
B = B/1e-4
ax1.plot(B, int_out1, 'b-', label='Port 1 (H)')
ax1.plot(B, int_out2, 'r-', label='Port 2 (V)')
ax1.plot(B, int_out1+int_out2, label='Total')
ax1.set_xlabel('Birefringence (1e-4)')
ax1.set_ylabel('Intensity (a.u.)')
ax1.set_ylim([0, 1.1])
ax1.set_xlim([min(B), max(B)])
ax1.set_xlim(ax1.get_xlim()[::-1])
ax1.set_xticks(np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], 5))
ax1.grid(True)

# Set the temperature x-axis limits and ticks
ax2.set_xlim(ax1.get_xlim())
ax2.plot(delta_t, delta_t)
ax2.set_xticks(np.linspace(ax2.get_xlim()[0], ax2.get_xlim()[1], 5))
ax2.set_xticklabels([f'{x:.1f}' for x in np.linspace(min(delta_t), max(delta_t), 5)])
ax2.set_xlabel('Temperature Change (C)')

ax1.legend(loc='upper left')
plt.tight_layout()

plt.show()