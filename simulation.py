"""Epidemiological model developed for TPD. Simulates an epidemic using an extended
SIR model with susceptible, exposed, infectious, symptomatic, and recovered buckets.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# TODO:
#   - Have E[i], I[i], and Sy[i] update right away so inf_tme and symp_time can be 0

def deterministic_SIRstep(pop, params, t):
    """Calculates one step of the SIR model. Steps are deterministic and fractional.
    
    ARGS: 
    params: list of model-specific parameters: 
        - beta: rate of infection
        - gamma: rate of recovery
        - inf_time: time it takes for an individual to become infectious
        - sympt_time: time it takes for an infectious individual to become symptomatic
        - N: population size
    pop:    values for each population bin at the current time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - R: recovered population
        - TotI: total population with the disease
    t: current time step

    RETURNS
    values for each population bin at the next time step, indexed as follows:
    Susceptible, Acquired, Infectious, Symptomatic, Recovered, E + I + S (Total Infections)
    """
    beta, gamma, inf_time, symp_time, N = params
    S, E, I, Sy, R, TotI = pop[t-1][0:6]

    # calculate changes in population bins
    new_acquired = (S*(I+Sy)*beta)/N
    new_recovered = Sy*gamma

    if t < inf_time: # prevents indexing error
        new_infectious = 0
    else:
        new_infectious = pop[t-inf_time][6]
    if t < symp_time: # prevents indexing error
        new_symptomatic = 0
    else:
        new_symptomatic = pop[t-symp_time][7]

    # update population bins
    S1 = S-new_acquired
    E1 = E+new_acquired-new_infectious
    I1 = I+new_infectious-new_symptomatic
    Sy1 = Sy+new_symptomatic-new_recovered
    R1 = R+new_recovered
    TotI1 = E1 + I1 + Sy1
    return [S1, E1, I1, Sy1, R1, TotI1, new_acquired, new_infectious]

def stochastic_SIRstep(pop, params, t):
    """Calculates one step of the SIR model. Steps are stochastic and discrete
    (no fractional populations)
    
    ARGS: 
    params: list of model-specific parameters: 
        - beta: rate of infection
        - gamma: rate of recovery
        - inf_time: time it takes for an individual to become infectious
        - sympt_time: time it takes for an infectious individual to become symptomatic
        - N: population size
    pop:    values for each population bin at the current time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - R: recovered population
        - TotI: total population with the disease
    t: current time step

    RETURNS
    values for each population bin at the next time step, indexed as follows:
    Susceptible, Acquired, Infectious, Symptomatic, Recovered, Total Infections (A + I + Sy)
    """
    beta, gamma, inf_time, symp_time, N = params
    S, E, I, Sy, R, TotI = pop[t-1][0:6]

    # calculate changes in population bins
    p_acquired = (beta*(I+Sy)*S)/(N*N)
    p_recovered = (Sy*gamma)/N
    new_acquired = np.random.binomial(S, p_acquired)
    new_recovered = np.random.binomial(Sy, p_recovered)
    if t < inf_time: # prevents indexing error
        new_infectious = 0
    else:
        new_infectious = pop[t-inf_time][6]
    if t < symp_time: # prevents indexing error
        new_symptomatic = 0
    else:
        new_symptomatic = pop[t-symp_time][7]

    # update population bins
    S1 = S-new_acquired
    E1 = E+new_acquired-new_infectious
    I1 = I+new_infectious-new_symptomatic
    Sy1 = Sy+new_symptomatic-new_recovered
    R1 = R+new_recovered
    TotI1 = E1 + I1 + Sy1
    return [S1, E1, I1, Sy1, R1, TotI1, new_acquired, new_infectious]


def simulate(pop, params, tmax, step_type="stochastic"):
    """Carries out a simulation of the model with the stated parameters,
    creating a pop numpy matrix that stores information about the simulated
    population for each time step
    
    ARGS:
    params: list of model-specific parameters: 
        - beta: rate of infection
        - gamma: rate of recovery
        - inf_time: time it takes for an individual to become infectious
        - sympt_time: time it takes for an infectious individual to become symptomatic
        - N: population size
    pop:    values for each population bin at the current time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - R: recovered population
        - TotI: total population with the disease
    tmax:   number of time steps to run the simulation
    step_type: determines whether steps should be calculated deterministically or stochastically.
        - Valid inputs: "stochastic" or "deterministic"

    RETURNS:
    pop:    updated pop numpy matrix
    """
    if step_type not in ["stochastic","deterministic"]:
        print("Error: You have input an invalid type for step_type. Please input either 'stochastic' or 'deterministic'.")
        return
    for i in range(1, tmax+1):
        if step_type == "stochastic":
            pop[i] = stochastic_SIRstep(pop, params, i)
        else:
            pop[i] = deterministic_SIRstep(pop, params, i)

    return pop

def plot_sim(pop):
    """Given a pop numpy matrix, plots relevant information about the population
    using matplotlib.
    
    ARGS:
    pop:    values for each population bin at the current time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - R: recovered population
        - TotI: total population with the disease

    RETURNS:
    none. Plots a graph for the following parameters of pop over time:
        - S
        - E
        - I
        - S
        - R
    """
    nrows, ncols = pop.shape
    t = np.linspace(0, nrows-1, nrows)
    N = sum(pop[0][0:5])

    plt.figure()
    plt.grid()
    plt.title("Epidemiological Model")
    plt.plot(t, pop[:,0], 'orange', label='Susceptible')
    plt.plot(t, pop[:,1], 'blue', label='Exposed')
    plt.plot(t, pop[:,2], 'r', label='Infectious')
    plt.plot(t, pop[:,3], 'g', label='Symptomatic')
    plt.plot(t, pop[:,4], 'yellow', label='Recovered')
    # plt.plot(t, pop[:,5], 'purple', label='Total Infected')
    plt.xlabel('Time t, [days]')
    plt.ylabel('Numbers of individuals')
    plt.ylim([0,N])
    plt.legend()

    plt.show()

def bloodnet(pop, blood_params):
    """Given a pop numpy matrix, calculates the probability of a bloodnet
    surveilance approach detecting the pathogen within the population
    
    ARGS:
    pop: values for each population bin at the current time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - R: recovered population
        - TotI: total population with the disease
    blood_params: model-specific parameters
        - detection_thresh_blood: number of infected donations required to detect pathogen
        - p_inf_donation: probability an infected person donates blood
        - p_donation_to_bloodnet: probability a donation occurs at a BloodNet center

    RETURNS:
    prob_detect: a [tmax+1,1] dimentional vector. For each time step,
    stores the probability of a bloodnet model detecting the pathogen.
    """
    detection_thresh_blood, p_inf_donation, p_donation_to_bloodnet = blood_params
    nrows, ncols = pop.shape
    tmax = nrows-1
    E = pop[:,1]
    I = pop[:,2]
    total_pos_trials = np.zeros(tmax+1) # total number of times an acquired/infected person could donate blood
    prob_detect = np.zeros(tmax+1)

    for i in range(1,tmax+1):
        total_pos_trials[i] = total_pos_trials[i-1]+E[i]+I[i]
        x = np.arange(0, detection_thresh_blood+1)
        cum_prob = binom.sf(x, total_pos_trials[i], p_inf_donation*p_donation_to_bloodnet)
        # print(f'\n {total_pos_trials[i]}, {cum_prob[detection_thresh]}')
        if total_pos_trials[i] <= detection_thresh_blood:
            prob_detect[i] = 0
        else:
            prob_detect[i] = cum_prob[detection_thresh_blood]

    return prob_detect

def threatnet(pop, threat_params):
    """Given a pop numpy matrix, calculates the probability of a threatnet
    surveilance approach detecting the pathogen within the population
    
    ARGS:
    pop: values for each population bin at the current time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - R: recovered population
        - TotI: total population with the disease
    threat_params: model-specific parameters
        - detection_thresh_threat: number of symptomatic patient sequences required to detect pathogen
        - p_inf_sequenced: probability that a symptomatic person will get sequenced

    RETURNS:
    prob_detect: a [tmax+1,1] dimentional vector. For each time step,
    stores the probability of a threatnet model detecting the pathogen.
    """
    detection_thresh_threat, p_inf_sequenced = threat_params
    nrows, ncols = pop.shape
    tmax = nrows-1
    Sy = pop[:,3]
    total_pos_visits = np.zeros(tmax+1) # total number of times an acquired/infected person could visit ED
    prob_detect = np.zeros(tmax+1)

    for i in range(1,tmax+1):
        total_pos_visits[i] = total_pos_visits[i-1]+Sy[i]
        x = np.arange(0, detection_thresh_threat+1)
        cum_prob = binom.sf(x, total_pos_visits[i], p_inf_sequenced)
        # print(f'\n {total_pos_trials[i]}, {cum_prob[detection_thresh]}')
        if total_pos_visits[i] <= detection_thresh_threat:
            prob_detect[i] = 0
        else:
            prob_detect[i] = cum_prob[detection_thresh_threat]
        
    return prob_detect


def plot_net(pop, blood_params, threat_params):
    """Given a pop numpy matrix, calls different net methods and plots the
    probability of each method detecting the pathogen over time.
    
    ARGS:
    pop:    values for each population bin at the current time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - R: recovered population
        - TotI: total population with the disease
        
    RETURNS:
    none. Plots each probability vector over time using matplotlib.
    """
    nrows, ncols = pop.shape
    t = np.linspace(0, nrows-1, nrows)
    blood_prob = bloodnet(pop, blood_params)
    threat_prob = threatnet(pop, threat_params)
    # p_inf: percent of people who are not susceptible (A, I, Sy, or R)
    p_inf = np.zeros(nrows)
    for i in range(nrows):
        p_inf[i] = 1-pop[i][0]/sum(pop[0][0:5])

    plt.figure()
    plt.grid()
    plt.title("BloodNet Model")
    # plt.plot(t, total_pos_trials, 'red', label='Cumulative Infected')
    plt.plot(t, blood_prob, 'red', label='Prob of Detect (Bloodnet)')
    plt.plot(t, threat_prob, 'green', label='Prob of Detect (Threatnet)')
    plt.plot(t, p_inf, 'blue', label='% with pathogen or recovered')
    plt.xlabel('Time t, [days]')
    plt.ylabel('Numbers of individuals')
    plt.ylim([0,1])
    plt.legend()

    plt.show()