"""Epidemiological model developed for TPD. Simulates an epidemic using an extended
SIR model with susceptible, exposed, infectious, symptomatic, and recovered buckets.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import math

# TODO:
#   - Make a population Class that stores the info instead of numpy arrays
#   - add somewhere saying inf_time and symp_time can't be 0

def deterministic_SIRstep(pop, params, community_params, t):
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

def stochastic_SIRstep(pop, params, community_params, t):
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
    community_params: values storing information about how the population is split into communities
        - num_communities: number of semi-isolated communities
        - movement_matrix: migration matrix for population movement between communities
    t: current time step

    RETURNS
    values for each population bin at the next time step, separated by community, 
    indexed as follows:
        - Susceptible, Acquired, Infectious, Symptomatic, Recovered, Total Infections (A + I + Sy)
    """
    beta, gamma, inf_time, symp_time, N = params
    num_communities, movement_matrix = community_params
    pop_t = np.zeros((num_communities+1, 8))

    for i in range(num_communities):
        S, E, I, Sy, R, TotI = pop[t-1][i+1][0:6]
        N_i = sum([S, E, I, Sy, R])

        # calculate changes in population bins
        p_acquired = (beta*(I+Sy)*S)/(N_i**2)
        p_recovered = (Sy*gamma)/N_i
        new_acquired = np.random.binomial(S, p_acquired)
        new_recovered = np.random.binomial(Sy, p_recovered)
        if t < inf_time: # prevents indexing error
            new_infectious = 0
        else:
            new_infectious = pop[t-inf_time][i+1][6]
        if t < symp_time: # prevents indexing error
            new_symptomatic = 0
        else:
            new_symptomatic = pop[t-symp_time][i+1][7]
        
        pop_moved = np.dot(N_i, movement_matrix[i])

        # update population bins
        S1 = S-new_acquired
        E1 = E+new_acquired-new_infectious
        I1 = I+new_infectious-new_symptomatic
        Sy1 = Sy+new_symptomatic-new_recovered
        R1 = R+new_recovered
        TotI1 = E1 + I1 + Sy1
        pop_t[i+1] = [S1, E1, I1, Sy1, R1, TotI1, new_acquired, new_infectious]
        pop_t[0] = [sum(x) for x in zip(pop_t[0], [S1, E1, I1, Sy1, R1, TotI1, new_acquired, new_infectious])]

    pop_t_moved = np.dot(movement_matrix, pop_t[1:])

    output = np.vstack((pop_t[0], pop_t_moved))

    return output

def simulate(pop, SIR_params, community_params, tmax, step_type="stochastic"):
    """Carries out a simulation of the model with the stated parameters,
    creating pop numpy matrices for each community that store information 
    about the simulated population/communities for each time step
    
    ARGS:
    params: list of model-specific parameters: 
        - beta: rate of infection
        - gamma: rate of recovery
        - inf_time: time it takes for an individual to become infectious
        - sympt_time: time it takes for an infectious individual to become symptomatic
        - N: population size
    pop:    values for each population bin at a given time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - R: recovered population
        - TotI: total population with the disease
    community_params: values storing information about how the population is split into communities
        - num_communities: number of semi-isolated communities
        - movement_matrix: migration matrix for population movement between communities
    tmax:   number of time steps to run the simulation
    step_type: determines whether steps should be calculated deterministically or stochastically.
        - Valid inputs: "stochastic" or "deterministic"

    RETURNS:
    pop_list: updated pop numpy matrices for each community. pop_list[0] contains population aggregate
    """
    N = SIR_params[4]

    if step_type not in ["stochastic","deterministic"]:
        print("Error: You have input an invalid type for step_type. Please input either 'stochastic' or 'deterministic'.")
        return
    for i in range(1, tmax+1):
        if step_type == "stochastic":
            pop[i] = stochastic_SIRstep(pop, SIR_params, community_params, i)
        else:
            pop[i] = deterministic_SIRstep(pop, SIR_params, community_params, i)

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
        - Sy
        - R
    """
    nrows, num_communities, ncols = pop.shape
    t = np.linspace(0, nrows-1, nrows)
    N = sum(pop[0][0][0:5])

    plt.figure()
    plt.grid()
    plt.title("Epidemiological Model")
    plt.plot(t, pop[:,:,0][:,0], 'orange', label='Susceptible')
    plt.plot(t, pop[:,:,1][:,0], 'blue', label='Exposed')
    plt.plot(t, pop[:,:,2][:,0], 'r', label='Infectious')
    plt.plot(t, pop[:,:,3][:,0], 'g', label='Symptomatic')
    plt.plot(t, pop[:,:,4][:,0], 'yellow', label='Recovered')
    plt.xlabel('Time t, [days]')
    plt.ylabel('Numbers of individuals')
    plt.ylim([0,N])
    plt.legend()

    plt.show()

def bloodnet(pop, blood_params, sequencing_params):
    """Given a pop numpy matrix, calculates the probability of a bloodnet
    surveilance approach detecting the pathogen within the population.

    BloodNet: every day, a certain proportion of people donate blood, which are then sequenced and
    tested. Positive tests (indicating presence of a pathogen) occur at some rate even when there 
    is no pathogen. When positive tests increase, the chance of a true positive goes up. At some
    point, the chance of a true positive is high enough to reject the null hypothesis of no true 
    positives.
    
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
    pop = pop[:,0:,][:,0] # isolates the aggregate population data from the population tensor
    p_donation, p_donation_to_bloodnet = blood_params
    true_positive_rate, false_positive_rate = sequencing_params
    nrows, ncols = pop.shape
    tmax = nrows-1
    N = sum(pop[0][0:5])
    prob_detect = np.zeros(tmax+1)

    for i in range(1,tmax+1):
        don_pop = N-pop[i][3]
        num_samples = np.random.binomial(don_pop, p_donation * p_donation_to_bloodnet)

        if num_samples == 0:
            prob_detect[i] = 0.5
        else:
            x = np.arange(0, num_samples+1)

            # TODO: the test below assumes that the null hypothesis is 0 true positive tests. 
            #   In reality, there are true positives without an exponentially growing pathogen
            p_x_positives_null = binom.sf(x, num_samples, false_positive_rate) #probability distribution of obsering at least x positive results given no true positives

            p_infected = (pop[i][1]+pop[i][2])/don_pop #probability one person is infected
            p_clean = 1 - p_infected
            p_positive = true_positive_rate*p_infected + false_positive_rate*p_clean #probability a sequencing test will return a positive result
            num_positive_tests = np.random.binomial(num_samples, p_positive)

            print(f'day: {i:<3}, don_pop: {round(don_pop):<7}, num_samples: {round(num_samples,6):<12}, num_positives: {round(num_positive_tests,3):<12}, E+I: {round(pop[i][1]+pop[i][2]):<12}, Sy: {round(pop[i][3]):<12}, prob_detect: {round(1-p_x_positives_null[round(num_positive_tests)],6):<7}')

            prob_detect[i] = 1-p_x_positives_null[num_positive_tests]

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

def bloodnet_community(pop, blood_params, sequencing_params):
    """Given a pop numpy matrix, calculates the probability of a bloodnet
    surveilance approach detecting the pathogen within the population.

    BloodNet: every day, a certain proportion of people donate blood, which are then sequenced and
    tested. Positive tests (indicating presence of a pathogen) occur at some rate even when there 
    is no pathogen. When positive tests increase, the chance of a true positive goes up. At some
    point, the chance of a true positive is high enough to reject the null hypothesis of no true 
    positives.
    
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
    nrows, num_communities, ncols = pop.shape
    tmax = nrows-1
    N = sum(pop[0][0:5])
    prob_detect = np.zeros(tmax+1)

    for i in range(1,tmax+1):
        don_pop = N-pop[i][3]
        num_samples = np.random.binomial(don_pop, p_donation * p_donation_to_bloodnet)
        x = np.arange(0, num_samples)

        # TODO: the test below assumes that the null hypothesis is 0 true positive tests. 
        #   In reality, there are true positives without an exponentially growing pathogen
        p_x_positives_null = binom.sf(x, num_samples, false_positive_rate) #probability distribution of obsering at least x positive results given no true positives

        p_infected = (pop[i][1]+pop[i][2])/don_pop #probability one person is infected
        p_clean = 1 - p_infected
        p_positive = true_positive_rate*p_infected + false_positive_rate*p_clean #probability a sequencing test will return a positive result
        num_positive_tests = np.random.binomial(num_samples, p_positive)

        # print(f'day: {i:<3}, don_pop: {don_pop:<7}, num_samples: {round(num_samples,6):<12}, num_positives: {round(num_positive_tests,3):<12}, E+I: {pop[i][1]+pop[i][2]:<12}, Sy: {pop[i][3]:<12}, prob_detect: {round(1-p_x_positives_null[round(num_positive_tests)],6):<7}')

        prob_detect[i] = 1-p_x_positives_null[round(num_positive_tests)]


    return prob_detect


def threatnet(pop, threat_params, sequencing_params):
    """Given a pop numpy matrix, calculates the probability of a threatnet
    surveilance approach detecting the pathogen within the population.

    ThreatNet: symptomatic people and people sick with other illnesses visit hospitals with 
    a fixed probability, some of whom will get sequenced. Positive tests (indicating presence 
    of a pathogen) occur at some rate even when there is no pathogen. When positive tests 
    increase, the chance of a true positive goes up. At some point, the chance of a true 
    positive is high enough to reject the null hypothesis of no true positives.
    
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
    nrows, num_communities, ncols = pop.shape
    tmax = nrows-1
    Sy = pop[:,3]
    N = sum(pop[0][0:5])
    # total_pos_visits = np.zeros(tmax+1) # total number of times an acquired/infected person could visit ED
    prob_detect = np.zeros(tmax+1)

    for i in range(1, tmax+1):
        num_sick = (N - Sy[i]) * background_sick_rate + Sy[i]
        num_sequenced = np.random.binomial(num_sick, p_sick_sequenced)
        x = np.arange(0, num_sequenced)

        p_x_positives_null = binom.sf(x, num_sequenced, false_positive_rate)

        p_infected = (Sy[i])/num_sick #probability one person is infected out of the group of sequenced people
        p_clean = 1 - p_infected
        p_positive = true_positive_rate*p_infected + false_positive_rate*p_clean #probability a sequencing test will return a positive result
        num_positive_tests = np.random.binomial(num_sequenced, p_positive)

        prob_detect[i] = 1-p_x_positives_null[round(num_positive_tests)]

        print(f'day: {i:<3}, num_sick: {num_sick:<7}, num_sequenced: {round(num_sequenced,6):<12}, num_positives: {round(num_positive_tests,3):<12}, E+I: {pop[i][1]+pop[i][2]:<12}, Sy: {pop[i][3]:<12}, prob_detect: {round(1-p_x_positives_null[round(num_positive_tests)],6):<7}')


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
    nrows, num_communities, ncols = pop.shape
    t = np.linspace(0, nrows-1, nrows)
    blood_prob = bloodnet(pop, blood_params, sequencing_params)
    # threat_prob = threatnet(pop, threat_params, sequencing_params)
    # p_inf: percent of people who are not susceptible (A, I, Sy, or R)
    
    pop = pop[:,0:,][:,0] # isolates the aggregate population data from the population tensor
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
    # plt.plot(t, threat_prob, 'green', label='Prob of Detect (Threatnet)')
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