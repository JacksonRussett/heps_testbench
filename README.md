# heps_testbench

Suite of a functions for simulating the Polarization-Frequency Hyper-entangled Photon Pair Source (HEPS) based on ___.

Sagnac-based configuration of the source that we wish to simulate.
``` 
________x______              
|              |   _________ Output 1
|______        |  |         
|_PPSF_|       PBS           
|              |  |_________ Output 2
|_______x______|            
```

## Installation

After cloning the repo, run the following in a ```cmd``` window opened from the project directory.

```
pip install -r requirements.txt
```

All dependencies should be installed from on the list in the txt file.

## Usage

Main files are ```Static_SimUpdateSept.py``` and ```Dynamic_SimUpdateSept.py``` for now. Run these files to generate results/figures.

Need to explain further...

## Project Structure

```heps_state.py``` - contains the fundemental function for implementing the model of the HEPS.\
```heps_static_sim.py``` - contains the all functions for simulating a source with static parameters (i.e. power split, brightness, etc. are constant wrt time).\
```heps_dynamic_sim.py``` - contains the all functions for simulating a source with dynamic parameters (i.e. power split, brightness, etc. are varying wrt time).\
```Static_SimUpdateSept.py``` - contains specific simulation configurations for the update presentation in Sept 2024.\
```Dynamic_SimUpdateSept.py``` - contains specific simulation configurations for the update presentation in Sept 2024. Running this script will create a ```results``` folder and populate it with ```.npz``` files.\

### Support Functions
```meas_stats.py``` - contains the all functions for simulating coincidence measurements on the HEPS.

#### pmf directory
```pmf_sim.py``` - contains function for simulating the transformation of a polarization state as it travels through several connected sections of polarization maintaining fiber (PMF).\
```misaligned_pmf_intensity.py``` - example of temperature the dependence of the output intensities of a polarizing beam splitter (PBS) when its input is coming from PMF.

### Others
```tests``` - contains the unit tests for various functions in the project.\
```old``` - contains original scripts recovered from the void...