"""Epidemiological model developed for TPD. Simulates an epidemic using an extended
SIR model with susceptible, exposed, infectious, symptomatic, and recovered buckets.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import math

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

def bloodnet(pop, blood_params, sequencing_params):
    """Given a pop numpy matrix, calculates the probability of a bloodnet
    surveilance approach detecting the pathogen within the population.

    BloodNet: every day, a certain proportion of people donate blood, which are then sequenced. 
    Based on the proportion of people infected, and the statistic power/false positive rate of
    the sequencing test, each positive sequencing result has some probability of being a true
    postiive. After enough positive sequencing results, there is a high enough chance one 
    positive result is a true positive, and the virus is detected.

    Positive tests occur at some rate when there is no pathogen. When positive tests increase,
    the chance of a true positive goes up. Calculate when the chance of a positive test is high
    enough given the test parameters.
    
    ARGS:
    pop: values for each population bin at the current time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - R: recovered population
        - TotI: total population with the disease
    blood_params: model-specific parameters
        - p_donation: probability that a donation-eligible person donates blood
        - p_donation_to_bloodnet: probability a donation occurs at a BloodNet center
    sequencing_params: statistical parameters of the sequencing test
        - true_positive_rate: proportion of sequences w/pathogen the test detects (also called statistical power)
        - false_positive_rate: proportion of sequences w/no pathogen the test says has pathogen

    RETURNS:
    prob_detect: a [tmax+1,1] dimentional vector. For each time step,
    stores the probability of a bloodnet model detecting the pathogen.
    """
    p_donation, p_donation_to_bloodnet = blood_params
    true_positive_rate, false_positive_rate = sequencing_params
    nrows, ncols = pop.shape
    tmax = nrows-1
    N = sum(pop[0][0:5])
    prob_detect = np.zeros(tmax+1)

    for i in range(1,tmax+1):
        don_pop = N-pop[i][3]
        num_samples = np.random.binomial(don_pop, p_donation * p_donation_to_bloodnet)
        x = np.arange(0, num_samples)

        #the test below assumes that the null hypothesis is 0 true positive tests. 
        #   In reality, there are true positives without an exponentially growing pathogen
        p_x_positives_null = binom.sf(x, num_samples, false_positive_rate) #probability distribution of obsering at least x positive results given no true positives

        p_infected = (pop[i][1]+pop[i][2])/don_pop #probability one person is infected
        p_clean = 1 - p_infected
        p_positive = true_positive_rate*p_infected + false_positive_rate*p_clean #probability a sequencing test will return a positive result
        # num_positive_tests = num_samples * p_positive
        num_positive_tests = np.random.binomial(num_samples, p_positive)

        print(f'day: {i:<3}, don_pop: {don_pop:<7}, num_samples: {round(num_samples,6):<12}, num_positives: {round(num_positive_tests,3):<12}, E+I: {pop[i][1]+pop[i][2]:<12}, Sy: {pop[i][3]:<12}, prob_detect: {round(1-p_x_positives_null[round(num_positive_tests)],6):<7}')

        prob_detect[i] = 1-p_x_positives_null[round(num_positive_tests)]


    return prob_detect

    ### Commented out: previous implementation of this function, saving in case we need to revert and I am lazy
    # for i in range(1,tmax+1):
    #     don_pop = N-pop[i][4] #population of people who can donate blood (symptomatic people can't donate blood)
    #     p_infected = (pop[i][5]-pop[i][4])/don_pop #probability one person is infected
    #     p_clean = 1 - p_infected
    #     num_inf_donations = p_infected * num_daily_donations
    #     p_positive = true_positive_rate*p_infected + false_positive_rate*p_clean #probability a sequencing test will return a positive result
    #     p_infected_if_positive = (true_positive_rate*p_infected)/p_positive #probability a positive testing result is a true positive
    #     num_positive_tests = num_daily_donations * p_positive
    #     x = np.arange(0, 21) # TODO: don't make 21 a hardcoded value
    #     p_x_infected = binom.sf(x, num_positive_tests, p_infected_if_positive) #probability of x true positives
    #     prob_detect[i] = p_x_infected[20]
    #     # print(f'day: {i:<3}, TotI: {pop[i][5]:<9}, #_inf_don: {round(num_inf_donations, 1):<6}, p_inf: {round(p_infected, 7):<9}, p_pos: {round(p_positive, 7):<9}, p_i_p: {round(p_infected_if_positive, 7):<9}, #_pos: {round(num_positive_tests,1):<6}, p_ten_i: {round(p_x_infected[9], 7):<10}')

def threatnet(pop, threat_params, sequencing_params):
    """Given a pop numpy matrix, calculates the probability of a threatnet
    surveilance approach detecting the pathogen within the population.

    ThreatNet: symptomatic individuals visit hospitals with a fixed probability.
    If enough symptomatic people visit hospitals, then the virus is detected.
    
    ARGS:
    pop: values for each population bin at the current time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - R: recovered population
        - TotI: total population with the disease
    threat_params: model-specific parameters
        - background_sick_rate: the proportion of people who are sick with a non-exponentially-growing pathogen
        - p_sick_sequenced: probability that a sick person will get sequenced (both symptomatic w/pathogen of interest and sick regularly)

    RETURNS:
    prob_detect: a [tmax+1,1] dimentional vector. For each time step,
    stores the probability of a threatnet model detecting the pathogen.
    """
    background_sick_rate, p_sick_sequenced = threat_params
    true_positive_rate, false_positive_rate = sequencing_params
    nrows, ncols = pop.shape
    tmax = nrows-1
    Sy = pop[:,3]
    # total_pos_visits = np.zeros(tmax+1) # total number of times an acquired/infected person could visit ED
    prob_detect = np.zeros(tmax+1)

    for i in range(1, tmax+1):
        prob_detect[i] = 0
        # sick_pop = pop[i][3] + (sum(pop[i])-pop[i][3]) * background_sick_rate
        # num_sick_sequenced = np.random.binomial(sick_pop, p_sick_sequenced)
        # x = np.arange(0, num_sick_sequenced)

        # p_x_positives_null = binom.sf(x, num_sick_sequenced, false_positive_rate) #probability distribution of obsering at least x positive results given no true positives

        # # wrong, fix
        # p_infected = pop[i][3]/sick_pop + ((pop[i][1] + pop[i][2]) * background_sick_rate)/sick_pop #probability one sick person is infected
        # p_clean = 1 - p_infected
        # p_positive = true_positive_rate*p_infected + false_positive_rate*p_clean #probability a sequencing test will return a positive result
        # num_positive_tests = sick_pop * p_positive

        # prob_detect[i] = 1-p_x_positives_null[round(num_positive_tests)]

        # print(f'day: {i:<3}, sick_pop: {round(sick_pop,1):<7}, num_samples: {round(num_sick_sequenced,6):<12}, p_infected: {round(p_infected,3):<12}, Sy: {pop[i][3]}, prob_detect: {round(1-p_x_positives_null[round(num_positive_tests)],6):<7}')


    print("done")

    # for i in range(1,tmax+1):
    #     # TODO: need to know the rate of sick people getting checked at hospitals
    #     num_samples = Sy[i] * p_inf_sequenced
    #     x = np.arange(0, num_samples)
    #     p_x_positives_null = binom.sf(x, num_samples, false_positive_rate)
        
    return prob_detect


def plot_net(pop, blood_params, threat_params, sequencing_params):
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
    blood_prob = bloodnet(pop, blood_params, sequencing_params)
    threat_prob = threatnet(pop, threat_params, sequencing_params)
    # p_inf: percent of people who are not susceptible (A, I, Sy, or R)
    p_inf = np.zeros(nrows)
    p_symp = np.zeros(nrows)
    for i in range(nrows):
        p_inf[i] = (pop[i][1]+pop[i][2])/sum(pop[0][0:5])
        p_symp[i] = (pop[i][3])/sum(pop[0][0:5])

    print("about to make graph")

    plt.figure()
    plt.grid()
    plt.title("BloodNet Model")
    plt.plot(t, blood_prob, 'red', label='Prob of Detect (Bloodnet)')
    plt.plot(t, threat_prob, 'green', label='Prob of Detect (Threatnet)')
    plt.plot(t, p_inf, 'blue', label='% with pathogen or recovered')
    plt.plot(t, p_symp, 'yellow', label='% symptomatic')
    plt.xlabel('Time t, [days]')
    plt.ylabel('Numbers of individuals')
    plt.ylim([0,1])
    plt.legend()

    plt.show()

# def fill_param_space(blood_params, threat_params, N, max_params):
#     detection_thresh_blood, p_inf_donation, p_donation_to_bloodnet = blood_params
#     detection_thresh_threat, p_inf_sequenced = threat_params
#     max_beta, max_inf, max_symp = max_params

#     #params: beta, gamma, inf_time, symp_time
#     beta_var = np.linspace(0, max_beta, max_beta*10+1)
#     inf_var = np.linspace(0,max_inf, max_inf+1)
#     inf_symp = np.linspace(0,max_symp, max_symp+1)

#     for i in beta_var:
#         for j in inf_var:
#             for k in inf_symp:
#                 params = [i, 0.1, j, k, N]
                

#     plt.figure()
#     plt.title("Parameter Space Coloring")
#     plt.add_subplot(projection='3d')
#     for i in range()

#     # plt.plot(t, total_pos_trials, 'red', label='Cumulative Infected')
#     plt.xlabel('Time t, [days]')
#     plt.ylabel('Numbers of individuals')
#     plt.ylim([0,1])
#     plt.legend()

#     plt.show()