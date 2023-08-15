# Pathogen Net Sim

Pathogen Net Sim is a SIR simulation used to model the spread of a novel pathogen starting from one individual in a homogenous, interconnected population. In addition to SIR modeling, NAME also contains a number of "detector function" that model various potential pathogen surveilance methods. It continuously calls these detector functions while the SIR model is running to provide information to the user about how quickly each surveilance method would detect the novel pathogen. 

Pathogen Net Sim currently requires python 3.10 to run correctly.

# Contents

- Motivation
- Getting Started
- Useage
- Adding Detector Functions
- Future Steps

## Motivation

Current methods to detect pandemics before they become widespread have historically failed. The the COVID pandemic showed that efforts to prevent the spread of future pandemics would save many lives and prevent the economic costs of future pandemics. Furthermore, pandemic detection prevents worst-case pathogens from causing even more damage than humanity experienced during COVID. This model aims to provide quantitative information about the detection speed of various pathogen surveilance methods, as well as a framework to enable testing additional surveilance methods not currently included in the model.

## Getting Started

This program is informally set up, meaning that it works by directly running files. Clone this repository to your local system, and then run the main.py file to get started. Main.py has a lot of commented-out code, which can be uncommented out to test various features of the model.

## Usage

The program contains two files: main.py and simulation.py. The functions of each are described below, along with an overview of how to use them.

### main.py

This file contains all of the user-specific information used to run one simulation. It is the file that you run for the program to output useful information. 

main.py contains three subsections:

- The first section, entitled `MUTABLE PARAMS, USER CAN CHANGE`, are all of the parameters of the model that the user can change depending on the type of population, pathogen, and pandemic response desired. Descriptions of each parameter are detailed in the description of the Population class in simulation.py.
- The second section, entitled `IMMUTABLE PARAMS, FINE-TUNED FOR ACCURACY`, are parameters specific to the detector functions already encoded into the program. These can be changed, but are already fine-tuned to best represent the real-world environment. Descriptions of each parameter are detailed in the description of the Population class in simulation.py.
- The final seciont, entitled, `EXAMPLE CODE FOR RUNNING THE SIMULATION`, are a series of example calls to functions in simulation.py that output useful information for testing various pathogen surveilance methods. More information on the specific calls already included in main.py are included in the docstrings of the individual functions of simulation.py.

### simulation.py

This file contains the code defining a Population class, along with all of the methods required to simulate a novel pathogen outbreak and evaluate the efficiency of surveilance methods. The methods are split up into five sections detailed below:

- The first section contains the constructor and all methods that update the class and instance variables of a Population.
- The second section, entitled `METHODS FOR RUNNING AND VISUALIZING THE SIR SIMULATION`, consist of all the methods that are used to simulate the outbreak of a novel pathogen within the `pop` matrix of a Population. Each method is detailed in their respective docstring, but in a nutshell: `simulate()` calls one of the two step functions each time step, updating the `pop` matrix sequentially. After `simulate()` is called, `plot_pop()` or `plot_sim()` can be called to visualize how the outbreak spreads over time in a matplotlib graph. In addition, `simulate()` also updates the likelihood of various detector functions detecting the pathogen each time step, allowing for later methods to analyze the efficiency of these various surveilance methods.
- The third section, entitled `METHODS FOR RUNNING AND VISUALIZING VARIOUS SURVEILANCE METHODS' EFFICIENCIES`, contains the detector funtions. Each detector function returns the likelihood of a specific surveilance method (eg: sequencing blood donations for signs of novel pathogens) detecting the novel pathogen for a specific timepoint in the simulation. These methods are then called by `simulate()` each step, creating an `tmax+1 x 1` matrix of probabilities representing the likelihood of detection over time. `plot_net()` plots these probability vectors over time to visualize the efficiency of each surveilance method.
- The fourth section, entitled `METHODS FOR RUNNING AND VISUALIZING PARAMETER SEARCHING`, provides methods that determine the best surveilance method across different parameter spaces of the model. For instance, `SIR_param_tester()` varies the values of the SIR params (detailed in the Population class description at the top of simulation.py), and determines which detector function most quickly detects the novel pathogen for each set of SIR params. These methods are useful for informing the user about the efficiency of different detector functions, and thus the efficiency of different surveilance methods in detecting a novel pathogen.
- The final section, entitled `METHODS FOR ESTIMATING COST SAVINGS OF SURVEILANCE METHODS`, contains methods that estimate the lives each detector function saves, assuming that the government institutes quarantine measures after the novel pathogen is detected. These methods use the `scaled_infectivity` class variable (contained within `lockdown_params`) to decrease infectivity after detection as a simulation of lockdown. If the methods from section four are meant to inform the user of how fast a survielance method detects a novel pathogen, these methods calculate the cost savings of faster detection for a given pathogen. Cost savings are currently calculated by multiplying the number of lives saved by the value of a statistical life (VSL). 

## Adding Detector Functions

While the model currently includes a number of detector functions, this list is by no means comprehensive. To add your own detector functions for specific surveilance methods, follow the steps below:

1. Create a new `tmax+1 x 1` numpy vector in the constructor, labeled appropriately. This will be the probability vector that stores your likelihood of detection at each timestep.
2. The detector function can be set up in any way you choose, but should be able to be called by `simulate()` at a given timestep, and should return nothing while updating the numpy vector from above.
3. Ensure that `simulate()` calls your detector function each timestep.
4. Ensure that `plot_net()` accurately plots your numpy probability vector.
5. Update the code in `test_nets()` to allow for it to determine the day of detection for your detector function. This will likely include:
    1. Creating an empty list like `blood_prob_list = []`
    2. Creating a temporary variable like `blood_prob = self.blood_prob` inside the multiplex loop, as well as appending the day of detection to your list like `blood_prob_list.append(self.day_of_detection(blood_prob))`
    3. Adding your prob_list to the bottom for loop and the output call, along with an acronym used in later methods
6. Update `legend_dict` in `SIR_param_tester` with the new acronym used above
7. Add another `elif` case to `lockdowm_param_tester()`
8. Update `prob_list` in `simulate_with_lockdown()` accordingly, as well as add another call to your detector function within the simulation part of `simulate_with_lockdown()`

Hopefully this will help with the addition of detector functions. I understand that this process may be lengthy and complicated, and future developers should consider streamlining this process.

## Future Steps

Below are a list of bugs and/or features that would improve the capabilities or ease-of-development of this model. This list is not comprehensive and future developers are encouraged to further contribute to this list.

- Currently, `inf_time` and `symp_time` cannot be 0 because the values for the current time step would be referenced, which are currently not updated.
- **Priority for developers: Simulation with multiple communities is currently broken and should be fixed as soon as possible. The current problem is that a dot product is used to move individuals between communities, but the values of `new_exposed` and `new_infectious` must also change to reflect exposed and infected individuals who moved between communities. This is not possible by simply taking the dot product of the current `pop` timestep with the `movement_matrix`. Future developers should either find a way to solve this problem or consider chaning `inf_time` and `symp_time` to rates instead of fixed time intervals. I have left commented-out code in simulation.py of my progress to try and fix this problem.**
- `deterministic_SIR_step()` is outdated and should be updated to be usable.
- Current detector functions calculate the likelihood of there being one true positive out of all the positive tests after sampling. This could be changed if it makes more sense to be stricter with detection.
- `plot_net()` should calculate a 7-day average for each detector function probability vector. This would make each plot look cleaner and allow for all probability vectors to be plotted on one graph.
- `astute_net()` is calculated fairly arbitrarily right now. More research could be done into determining how fast doctors figure out a disease exists without surveilance systems in place.
- All of the methods in `METHODS FOR ESTIMATING COST SAVINGS OF SURVEILANCE METHODS` should be run multiple times and their outputs averaged for more consistent results. The current results can vary by up to a factor of 2, which suggests high variance. Ideally, future developers should find a way to multiplex and capture the variance of the simulations.
