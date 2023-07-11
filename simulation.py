"""Epidemiological model developed for TPD. Simulates an epidemic using an extended
SIR model with susceptible, exposed, infectious, symptomatic, and recovered buckets.
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import binom
import math
import pandas as pd
from functools import reduce
import copy

# TODO:
#   - add somewhere saying inf_time and symp_time can't be 0

class Population():
    """Defines a population of individuals during a novel pandemic. Encodes relevant
    parameters and has a number of functions for SIR modeling and pathogen detection 
    analysis.

    IMPORTANT PARAMETERS:
    N: population size
    tmax: time simulation will run for
    t: tmax x 1 numpy array for plotly graphs
    pop: tmax x num_communities x 8 numpy array. Values for each population bin at the current time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - Asy: asymptoamtic population
        - R: recovered population
        - D: dead population
        - TotI: total population with the disease
        - new_exposed: newly exposed individuals in the population
        - new_infectious: newly infected individuals in the population
    community_params: values storing information about how the population is split into communities
        - num_communities (n): number of semi-isolated communities
        - initial_community_sizes: n x 1 numpy array. Initial relative sizes of each community
        - movement_matrix: n x n numpy array. Migration matrix for population movement between communities
    sequencing_params: statistical parameters of the sequencing test being used
        - true_positive_rate: proportion of sequences w/pathogen the test detects (also called statistical power)
        - false_positive_rate: proportion of sequences w/no pathogen the test says has pathogen
    blood_params: model-specific parameters for the BloodNet surveilance system
        - p_donation: probability that a donation-eligible person donates blood
        - p_donation_to_bloodnet: probability a donation occurs at a BloodNet center
    threat_params: model-specific parameters for the ThreatNet surveilance system
        - background_sick_rate: the proportion of people who are sick with a non-exponentially-growing pathogen
        - p_sick_sequenced: probability that a sick person will get sequenced (both symptomatic w/pathogen of interest and sick regularly)
    astute_params: pmodel-specific parameters for the AstuteNet surveilance system
        - p_hospitalized: the proportion of symptomatic people who go to a hospital
        - p_doctor_detect: the probability that a doctor reports a symptomatic case as a new pathogen
        - command_readiness: likelihood of a doctor's report being picked up by the system
    SIR_params: list of model-specific parameters: 
        - beta: rate of infection
        - gamma: rate of recovery
        - inf_time: time it takes for an exposed individual to become infectious
        - sympt_time: time it takes for an infectious individual to become symptomatic
        - p_asymp: probability that a symptomatic individual will not experience symptoms
        - mu: rate of death of symptomatic people
    """
    def __init__(self, N = 100000, initial_infected = 1, tmax = 150, community_params = [1, [1], [1]]):
        self.N = N
        self.tmax = tmax
        self.community_params = community_params
        num_communities, initial_community_sizes, movement_matrix = self.community_params

        self.pop = np.zeros((self.tmax+1, num_communities+1, 10))
        self.pop[0][0] = [N-1, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        for i in range(num_communities):
            infected_i = initial_infected if i == 0 else 0
            N_i = round(N*initial_community_sizes[i])
            pop_i = [N_i-infected_i, 0, infected_i, 0, 0, 0, 0, infected_i, 0, 0]
            self.pop[0][i+1] = pop_i
        
        self.t = np.linspace(0, tmax, tmax+1)
        self.blood_prob = np.zeros(self.tmax+1)
        self.threat_prob = np.zeros(self.tmax+1)
        self.astute_prob = np.zeros(self.tmax+1)
        self.community_blood_probs = np.zeros((num_communities, self.tmax+1))

    def set_all_parameters(self, sequencing_params, blood_params, threat_params, astute_params, SIR_params, lockdown_params):
        self.sequencing_params = sequencing_params
        self.blood_params = blood_params
        self.threat_params = threat_params
        self.astute_params = astute_params
        self.SIR_params = SIR_params
        self.lockdown_params = lockdown_params

    def set_detection_params(self, detection_params):
        self.threshold, self.time_delay = detection_params

    def set_SIR_params(self, SIR_params):
        self.SIR_params = SIR_params

    # METHODS FOR RUNNING AND VISUALIZING THE SIR SIMULATION

    def simulate(self, step_type="stochastic"):
        """Carries out a simulation of the model with the stated parameters,
        creating pop numpy matrices for each community that store information 
        about the simulated population/communities for each time step
        
        ARGS:
        step_type: determines whether steps should be calculated deterministically or stochastically.
            - Valid inputs: "stochastic" or "deterministic"

        RETURNS:
        None. Updates pop to reflect the result of the SIR simulation
        """
        if step_type not in ["stochastic","deterministic"]:
            print("Error: You have input an invalid type for step_type. Please input either 'stochastic' or 'deterministic'.")
            return
        for i in range(1, self.tmax+1):
            if step_type == "stochastic":
                self.pop[i] = self.stochastic_SIRstep(self.pop, 
                                                      self.SIR_params, 
                                                      self.community_params, 
                                                      i)    
            else:
                self.pop[i] = self.deterministic_SIRstep(self.pop, 
                                                         self.SIR_params, 
                                                         self.community_params, 
                                                         i)
            self.bloodnet(i)
            self.bloodnet_community(i)
            self.threatnet(i)
            self.astutenet(i)

        return

    def deterministic_SIRstep(self, pop, params, community_params, t):
        """Calculates one step of the SIR model. Steps are deterministic and fractional.
        
        ARGS: 
        pop: 3D numpy array. Values for each population bin at the current time step
            - S: susceptible population
            - E: exposed population with the disease who aren't infectious
            - I: infectious population that isn't symptomatic
            - Sy: symptomatic population
            - R: recovered population
            - D: dead population
            - TotI: total population with the disease
            - new_exposed: newly exposed individuals in the population
            - new_infectious: newly infected individuals in the population
        params: list of model-specific parameters: 
            - beta: rate of infection
            - gamma: rate of recovery
            - inf_time: time it takes for an individual to become infectious
            - sympt_time: time it takes for an infectious individual to become symptomatic
            - N: population size
        t: current time step

        RETURNS
        values for each population bin at the next time step, indexed as follows:
        Susceptible, Exposed, Infectious, Symptomatic, Recovered, E + I + S (Total Infections)
        """
        beta, gamma, inf_time, symp_time, p_asymp, mu = params
        S, E, I, Sy, Asy, R, D, TotI = pop[t-1][0:8]

        # calculate changes in population bins
        new_exposed = (S*(I+Sy+Asy)*beta)/self.N
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
        S1 = S-new_exposed
        E1 = E+new_exposed-new_infectious
        I1 = I+new_infectious-new_symptomatic
        Sy1 = Sy+new_symptomatic-new_recovered
        R1 = R+new_recovered
        TotI1 = E1 + I1 + Sy1
        
        return [S1, E1, I1, Sy1, R1, TotI1, new_exposed, new_infectious]

    def stochastic_SIRstep(self, pop, params, community_params, t):
        """Calculates one step of the SIR model. Steps are stochastic and discrete
        (no fractional populations)
        
        ARGS:
        pop: 3D numpy array. Values for each population bin at the current time step
            - S: susceptible population
            - E: exposed population with the disease who aren't infectious
            - I: infectious population that isn't symptomatic
            - Sy: symptomatic population
            - R: recovered population
            - D: dead population
            - TotI: total population with the disease
            - new_exposed: newly exposed individuals in the population
            - new_infectious: newly infected individuals in the population 
        params: list of model-specific parameters: 
            - beta: rate of infection
            - gamma: rate of recovery
            - inf_time: time it takes for an individual to become infectious
            - sympt_time: time it takes for an infectious individual to become symptomatic
            - N: population size
        community_params: values storing information about how the population is split into communities
            - num_communities: number of semi-isolated communities
            - movement_matrix: migration matrix for population movement between communities
        t: current time step

        RETURNS
        values for each population bin at the next time step, separated by community, 
        indexed as follows:
            - Susceptible, Exposed, Infectious, Symptomatic, Recovered, Total Infections (E + I + Sy)
        """
        beta, gamma, inf_time, symp_time, p_asymp, mu = params
        num_communities, initial_community_sizes, movement_matrix = community_params
        pop_t = np.zeros((num_communities+1, 10))

        for i in range(num_communities):
            S, E, I, Sy, Asy, R, D, TotI = pop[t-1][i+1][0:8]
            N_i = sum([S, E, I, Sy, Asy, R, D])

            # calculate changes in population bins
            p_exposed = (beta*(I+Sy+Asy))/N_i
            new_exposed = np.random.binomial(S, p_exposed)
            new_Sy_recovered = np.random.binomial(Sy, gamma)
            new_Asy_recovered = np.random.binomial(Asy, gamma)
            new_dead = np.random.binomial(Sy, mu)
            if t < inf_time: # prevents indexing error
                new_infectious = 0
            else:
                new_infectious = pop[t-inf_time][i+1][8]
            if t < symp_time: # prevents indexing error
                new_symptomatic = 0
                new_asymptomatic = 0
            else:
                new_symptomatic = np.random.binomial(pop[t-symp_time][i+1][9], 1-p_asymp)
                new_asymptomatic = pop[t-symp_time][i+1][9] - new_symptomatic
            
            pop_moved = np.dot(N_i, movement_matrix[i])

            # update population bins
            S1 = S-new_exposed
            E1 = E+new_exposed-new_infectious
            I1 = I+new_infectious-new_symptomatic-new_asymptomatic
            Sy1 = Sy+new_symptomatic-new_Sy_recovered-new_dead
            Asy1 = Asy+new_asymptomatic-new_Asy_recovered
            R1 = R+new_Sy_recovered+new_Asy_recovered
            D1 = D+new_dead
            TotI1 = E1 + I1 + Sy1 + Asy1
            pop_t[i+1] = [S1, E1, I1, Sy1, Asy1, R1, D1, TotI1, new_exposed, new_infectious]
            pop_t[0] = [sum(x) for x in zip(pop_t[0], [S1, E1, I1, Sy1, Asy1, R1, D1, TotI1, new_exposed, new_infectious])]

        # TODO: multiple communities is broken and probably can't be fixed unless I change inf_time and symp_time to rates

        pop_t_moved = np.dot(movement_matrix, pop_t[1:])

        output = np.vstack((pop_t[0], pop_t_moved))

        return output

    def plot_sim(self, pop, title):
        """Given a pop numpy matrix, plots relevant information about the population
        using matplotlib.
        
        ARGS:
        none.

        RETURNS:
        none. Plots a graph for the following parameters of pop over time:
            - S
            - E
            - I
            - Sy
            - Asy
            - R
            - D
        """
        plt.figure()
        plt.grid()
        plt.title(title)
        plt.plot(self.t, pop[:,:,0][:,0], 'orange', label='Susceptible')
        plt.plot(self.t, pop[:,:,1][:,0], 'blue', label='Exposed')
        plt.plot(self.t, pop[:,:,2][:,0], 'r', label='Infectious')
        plt.plot(self.t, pop[:,:,3][:,0], 'g', label='Symptomatic')
        plt.plot(self.t, pop[:,:,4][:,0], 'purple', label='Asymptomatic')
        plt.plot(self.t, pop[:,:,5][:,0], 'yellow', label='Recovered')
        plt.plot(self.t, pop[:,:,6][:,0], 'black', label='Dead')
        plt.xlabel('Time t, [days]')
        plt.ylabel('Numbers of individuals')
        plt.ylim([0,self.N])
        plt.legend()

        plt.show()

    # METHODS FOR RUNNING AND VISUALIZING VARIOUS SURVEILANCE METHODS' EFFICIENCIES

    def bloodnet(self, t, community_i=0):
        """Given a pop numpy matrix, calculates the probability of a bloodnet
        surveilance approach detecting the pathogen within the population.

        BloodNet: every day, a certain proportion of people donate blood, which are then sequenced and
        tested. Positive tests (indicating presence of a pathogen) occur at some rate even when there 
        is no pathogen. When positive tests increase, the chance of a true positive goes up. At some
        point, the chance of a true positive is high enough to reject the null hypothesis of no true 
        positives.
        
        ARGS:
        t: time step value
        community_i: instructions for which community to use for calculations. Default is total population (0).

        RETURNS:
        None. For each time step, stores the probability of a threatnet model
        detecting the pathogen in self.blood_prob
        """
        temp_pop = self.pop[:,0:,][:,community_i] # isolates the desired population data from the population tensor
        p_donation, p_donation_to_bloodnet = self.blood_params
        true_positive_rate, false_positive_rate = self.sequencing_params

        don_pop = self.N-temp_pop[t][3]-temp_pop[t][6]
        infected_pop = temp_pop[t][1]+temp_pop[t][2]+temp_pop[t][4]
        num_samples = np.random.binomial(don_pop, p_donation * p_donation_to_bloodnet)

        if num_samples == 0:
            self.blood_prob[t] = 0.5
        else:
            x = np.arange(0, num_samples+1)

            # TODO: the test below assumes that the null hypothesis is 0 true positive tests. 
            #   In reality, there are true positives without an exponentially growing pathogen
            p_x_positives_null = binom.sf(x, num_samples, false_positive_rate) #probability distribution of obsering at least x positive results given no true positives

            p_infected = infected_pop/don_pop #probability one person is infected
            p_clean = 1 - p_infected
            p_positive = true_positive_rate*p_infected + false_positive_rate*p_clean #probability a sequencing test will return a positive result
            num_positive_tests = np.random.binomial(num_samples, p_positive)

            # print(f'community: {community_i}, day: {i:<3}, don_pop: {round(don_pop):<7}, num_samples: {round(num_samples,6):<12}, num_positives: {round(num_positive_tests,3):<12}, E+I: {round(pop[i][1]+pop[i][2]):<12}, Sy: {round(pop[i][3]):<12}, prob_detect: {round(1-p_x_positives_null[round(num_positive_tests)],6):<7}')

            self.blood_prob[t] = 1-p_x_positives_null[num_positive_tests]                

    def bloodnet_community(self, t):
        """Calls the bloodnet detection model on each sub-community
        of the population.
        
        ARGS:
        none.

        RETURNS:
        prob_detects: a [num_communities, tmax+1] dimentional vector. For each time step,
        stores the probability of a bloodnet model detecting the pathogen for each community.
        """
        num_communities, initial_community_sizes, movement_matrix = self.community_params

        for i in range(num_communities):
            temp_pop = self.pop[:,0:,][:,i] # isolates the desired population data from the population tensor
            N_i = sum(temp_pop[t][0:7])

            p_donation, p_donation_to_bloodnet = self.blood_params
            true_positive_rate, false_positive_rate = self.sequencing_params

            don_pop = N_i-temp_pop[t][3]-temp_pop[t][6]
            infected_pop = temp_pop[t][1]+temp_pop[t][2]+temp_pop[t][4]
            num_samples = np.random.binomial(don_pop, p_donation * p_donation_to_bloodnet)

            if num_samples == 0:
                self.community_blood_probs[i][t] = 0.5
            else:
                x = np.arange(0, num_samples+1)

                # TODO: the test below assumes that the null hypothesis is 0 true positive tests. 
                #   In reality, there are true positives without an exponentially growing pathogen
                p_x_positives_null = binom.sf(x, num_samples, false_positive_rate) #probability distribution of obsering at least x positive results given no true positives

                p_infected = infected_pop/don_pop #probability one person is infected
                p_clean = 1 - p_infected
                p_positive = true_positive_rate*p_infected + false_positive_rate*p_clean #probability a sequencing test will return a positive result
                num_positive_tests = np.random.binomial(num_samples, p_positive)

                # print(f'community: {community_i}, day: {i:<3}, don_pop: {round(don_pop):<7}, num_samples: {round(num_samples,6):<12}, num_positives: {round(num_positive_tests,3):<12}, E+I: {round(pop[i][1]+pop[i][2]):<12}, Sy: {round(pop[i][3]):<12}, prob_detect: {round(1-p_x_positives_null[round(num_positive_tests)],6):<7}')
                self.community_blood_probs[i][t] = 1-p_x_positives_null[num_positive_tests]

    def threatnet(self, t, community_i=0):
        """Given a pop numpy matrix, calculates the probability of a threatnet
        surveilance approach detecting the pathogen within the population.

        ThreatNet: symptomatic people and people sick with other illnesses visit hospitals with 
        a fixed probability, some of whom will get sequenced. Positive tests (indicating presence 
        of a pathogen) occur at some rate even when there is no pathogen. When positive tests 
        increase, the chance of a true positive goes up. At some point, the chance of a true 
        positive is high enough to reject the null hypothesis of no true positives.
        
        ARGS:
        community_i: instructions for which community to use for calculations. Default is total population

        RETURNS:
        None. For each time step, stores the probability of a threatnet model
        detecting the pathogen in self.threat_prob
        """
        temp_pop = self.pop[:,0:,][:,community_i] # isolates the desired population data from the population tensor
        background_sick_rate, p_hospitalized, p_hospital_sequenced = self.threat_params
        true_positive_rate, false_positive_rate = self.sequencing_params

        Sy = temp_pop[:,3]
        D = temp_pop[:,6]

        num_sick = (self.N - Sy[t] - D[t]) * background_sick_rate + Sy[t]
        num_sequenced = np.random.binomial(num_sick, p_hospitalized*p_hospital_sequenced)

        # If 10% of the population is sick, assume the pathogen has been detected.
        # Added this in to speed up the computation, it is definitely an assumption though
        x = np.arange(0, num_sequenced+1)

        p_x_positives_null = binom.sf(x, num_sequenced, false_positive_rate)

        p_infected = (Sy[t])/num_sick #probability one person is infected out of the group of sequenced people
        p_clean = 1 - p_infected
        p_positive = true_positive_rate*p_infected + false_positive_rate*p_clean #probability a sequencing test will return a positive result
        num_positive_tests = np.random.binomial(num_sequenced, p_positive)

        self.threat_prob[t] = 1-p_x_positives_null[num_positive_tests]

    def astutenet(self, t, community_i=0):
        """Given a pop numpy matrix, calculates the probability of an astute doctor
        successfully discovering and reporting a novel pathogen.

        When people get sick, they sometimes go to the hospital. These people are
        seen by doctors, and if enough people have novel symptoms then an astute
        doctor might realize there is something going on. If the doctor raises
        the alarm and their alarm is heard, then the new pathogen is detected.

        ARGS:
        community_i: instructions for which community to use for calculations. Default is total population

        RETURNS:
        None. For each time step, stores the probability of a threatnet model
        detecting the pathogen in self.astute_prob
        """
        p_hospitalized, p_doctor_detect, chi = self.astute_params
        temp_pop = self.pop[:,0:,][:,community_i] # isolates the desired population data from the population tensor

        Sy = temp_pop[:,3]

        num_symp_hospitalized = np.random.binomial(Sy[t], p_hospitalized)
        num_hospital_reports = np.random.binomial(num_symp_hospitalized, p_doctor_detect)

        # not stochastic, can make it stochastic w/a cdf. This function is very arbitrary and BOTEC
        p_investigation = 1 / (1 + (50/chi) * math.e**(-2 * chi * num_hospital_reports))

        self.astute_prob[t] = p_investigation

    def wastenet(self, t, community_i=0):
        """Given a pop numpy matrix, calculates the probability of a wastewater
        sequencing system detecting a novel pathogen.

        Every day, wastewater is collected from filtration plants. This wastewater
        contains some viral particles from fecal shedding, which is then detected
        via sequencing processed wastewater samples. Positive sequencing results may
        occur even when no pathogen is present. When positive tests increase, the 
        chance of a true positive goes up. At some point, the chance of a true 
        positive is high enough to reject the null hypothesis of no true positives.

        ARGS:
        community_i: instructions for which community to use for calculations. Default is total population

        RETURNS:
        none.
        """

        temp_pop = self.pop[:,0:,][:,community_i] # isolates the desired population data from the population tensor
        return

    def plot_net(self):
        """Given a pop numpy matrix, calls different net methods and plots the
        probability of each method detecting the pathogen over time.
        
        ARGS:
        none.

        RETURNS:
        none. Plots each probability vector over time using pyplot.
        """
        nrows, nstacks, ncols = self.pop.shape
        num_communities = nstacks-1
        
        temp_pop = self.pop[:,0:,][:,0] # isolates the aggregate population data from the population tensor
        p_inf = np.zeros(nrows) # p_inf: percent of people who are not susceptible (E, I, Sy, or R)
        p_symp = np.zeros(nrows)
        for i in range(nrows):
            p_inf[i] = (temp_pop[i][1]+temp_pop[i][2])/sum(temp_pop[0][0:5])
            p_symp[i] = (temp_pop[i][3])/sum(temp_pop[0][0:5])

        # Astute Doctor Total Model
        plt.figure()
        plt.title(f'Astute Doctor Total Population Model')
        plt.plot(self.t, self.astute_prob, 'red', label='Prob of Detect (AstuteNet)')
        plt.plot(self.t, p_inf, 'blue', label='% infected/recovered')
        plt.plot(self.t, p_symp, 'yellow', label='% symptomatic')
        plt.xlabel('Time t, [days]')
        plt.ylabel('Numbers of individuals')
        plt.ylim([0,1])
        plt.legend()
        plt.show() 

        # Threatnet Total Population Model
        plt.figure()
        plt.title("ThreatNet Total Population Model")
        plt.plot(self.t, self.threat_prob, 'green', label='Prob of Detect (ThreatNet)')
        plt.plot(self.t, p_inf, 'blue', label='% infected/recovered')
        plt.plot(self.t, p_symp, 'yellow', label='% symptomatic')
        plt.xlabel('Time t, [days]')
        plt.ylabel('Numbers of individuals')
        plt.ylim([0,1])
        plt.legend()
        plt.show()  

        # Bloodnet Total Model
        plt.figure()
        plt.title("BloodNet Total Population Model")
        plt.plot(self.t, self.blood_prob, 'red', label='Prob of Detect (BloodNet)')
        plt.plot(self.t, p_inf, 'blue', label='% infected/recovered')
        plt.plot(self.t, p_symp, 'yellow', label='% symptomatic')
        plt.xlabel('Time t, [days]')
        plt.ylabel('Numbers of individuals')
        plt.ylim([0,1])
        plt.legend()
        plt.show()  

        # Bloodnet Community Models
        for i in range(num_communities):
            color = list(np.random.choice(range(256), size=3))
            plt.figure()
            plt.title(f'BloodNet Community {i+1} Model')
            plt.plot(self.t, self.community_blood_probs[i], color, label=f'Prob of Detect (BloodNet C{i+1})')
            plt.plot(self.t, p_inf, 'blue', label='% infected/recovered')
            plt.plot(self.t, p_symp, 'yellow', label='% symptomatic')
            plt.xlabel('Time t, [days]')
            plt.ylabel('Numbers of individuals')
            plt.ylim([0,1])
            plt.legend()
            plt.show()  

    # METHODS FOR RUNNING AND VISUALIZING PARAMETER SEARCHING

    def day_of_detection(self, prob_detect):
        """Given the output of a surveilance method (prob_detect), calculate
        the day when the method detects the pathogen. This is defined as the
        day when the probability of detection exceeds the threshold for seven conscutive
        days.

        ARGS:
        prob_detect: a tmax x 1 numpy array storing the probability the method detects a pathogen for a given day
        threshold: probability threshold that must be exceeded
        time_delay: the number of days prob_detect must exceed the threshold for the method to detect the pathogen
        
        RETURNS:
        day_of_detection: the day the method detects the pathogen, as defined above.
        """
        def threshold_checker(x):
            return 1 if x > self.threshold else 0

        v_thresh = np.vectorize(threshold_checker)
        prob_detect = v_thresh(prob_detect)

        for i in range(len(prob_detect)):
            if 0 not in prob_detect[i:i+self.time_delay]:
                return i+self.time_delay
        
        return self.tmax + 1
    
    def test_nets(self, num_runs=5):
        """Given a simulated population, calculates the day
        when each surveilance system detects the pathogen.

        ARGS:
        num_runs: number of multiplexed runs to average over

        RETURNS:
        output: list of tuples. Each tuple contains an acronym for a surveilance system and the day the system detects the outbreak
        thresh_dict[best_net]: acronym of the best-performing model
        best_net: the day of detection for the best-performing model
        """
        blood_prob_list = []
        threat_prob_list = []
        astute_prob_list = []
        for i in range(num_runs):
            self.simulate()
            blood_prob = self.blood_prob
            # community_blood_probs = self.bloodnet_community()
            astute_prob = self.astute_prob
            threat_prob = self.threat_prob

            blood_prob_list.append(self.day_of_detection(blood_prob))
            threat_prob_list.append(self.day_of_detection(threat_prob))
            astute_prob_list.append(self.day_of_detection(astute_prob))

        list_averages = []
        for list in [blood_prob_list, threat_prob_list, astute_prob_list]:
            list_averages.append(sum(list)/len(list))

        return [["B", list_averages[0]], ["T", list_averages[1]], ["A", list_averages[2]]]
    
    def sequencing_param_tester(self, num_runs=5):
        """Runs multiple SIR simulations using varied sequencing
        parameters, and determines the fastest surveilance method 
        for each set of parameters. Then visualizes which surveilance
        method is best for each set of sequencing parameters

        ARGS:
        num_runs: number of multiplexed runs to average over

        RETURNS:
        none. Plots a heatmap of surveilance method performance over
        parameter-space.
        """
        best_net_list = np.chararray((10, 9 ), unicode=True)
        best_net_values = np.zeros((10, 9))
        for i in range(10):
            for j in range(9):
                print(i, j)
                self.set_all_parameters([i/10, (j+1)/10], self.blood_params, self.threat_params, self.astute_params, self.SIR_params)
                test_nets = self.test_nets(num_runs)
                best_net_list[i][j], best_net_values[i][j] = reduce(lambda x, y: x if x[1] < y[1] else y, test_nets)
        
        self.sequencing_heatmap = sns.heatmap(best_net_values, linewidth=0.5, annot=best_net_list, fmt="")
        plt.title("Heatmap of Detection Methods with Varied Sequencing_Param")
        plt.xlabel('True_positive_rate')
        plt.ylabel('False_positive_rate')
        plt.show()

    def SIR_param_tester(self):
        """Runs multiple SIR simulations using varied SIR parameters, and
        determines the fastest surveilance method for each set of parameters.
        Then visualizes which surveilance method is best for each set of 
        SIR parameters

        ARGS:
        none. 

        RETURNS:
        none. Plots a 3D scatterplot of the best surveilance method for each
        set of SIR parameters. x/y/z axes are beta, gamma, and incubation time.
        Size of dot represents time to symptoms after incubation, and color
        represents the best-performing surveilance method given a set of params.
        """
        V = 7
        num_runs = 3
        best_net_list = np.chararray((V, V, V, V), unicode=True)
        best_net_values = np.zeros((V, V, V, V))
        best_net_relative_performance = np.zeros((V, V, V, V))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        color_dict = {"T": "green", "B": "red", "A": "blue"}
        legend_dict = {"T": "threatnet", "B": "bloodnet", "A": "astute doctor"}

        beta_values = [0.25, 0.5, 1, 1.5, 2, 3, 4]
        gamma_values = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
        inf_time_values = [1, 5, 9, 13, 17, 21, 25]
        symp_time_values = [1, 5, 9, 13, 17, 21, 25]
        sympt_time_placeholder = 5
        p_asymp_values = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9]

        for i in range(V):
            for j in range(V):
                for k in range(V):
                    for l in range(V):
                        beta, gamma, inf_time, p_asymp = [beta_values[i], gamma_values[j], inf_time_values[k], p_asymp_values[l]]
                        print(beta, gamma, inf_time, p_asymp)
                        self.set_all_parameters(self.sequencing_params, self.blood_params, self.threat_params, self.astute_params, [beta, gamma, inf_time, sympt_time_placeholder, p_asymp])

                        # store the best-performing
                        test_nets = self.test_nets(num_runs)
                        best_net_list[i, j, k, l], best_net_values[i, j, k, l] = reduce(lambda x, y: x if x[1] < y[1] else y, test_nets)

                        net_values = sorted(set(val for name, val in test_nets))
                        second_best_net_value = net_values[-1]
                        best_net_relative_performance[i, j, k, l] =  second_best_net_value - best_net_values[i, j, k, l]

                        ax.scatter(beta, gamma, inf_time-l/2, s=p_asymp*20, alpha=best_net_relative_performance[i, j, k, l]/(self.tmax+1), c=color_dict[best_net_list[i, j, k, l]], marker='o', label=legend_dict[best_net_list[i, j, k, l]])

        plt.title("Scatterplot of Detection Methods with Varied SIR_params")
        ax.set_xlabel('infection rate')
        ax.set_ylabel('recovery rate')
        ax.set_zlabel('incubation period')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title="Best surveilance method")
        plt.show()

    def lockdowm_param_tester(self, num_runs=5):
        """Runs multiple lockdown SIR simulations varying percent_reduced_infectivity,
        then determines how fast each surveilance method detects the pathogen. Then,
        calls simulate_with_lockdown to estimate the change in deaths between scenarios
        wtih each surveilance method implemented, and visualizes deaths for each model
        over the percent_reduced_infectivity space [0,1].

        ARGS:
        num_rums: number of runs to multiplex and average over 

        RETURNS:
        none. Plots a 2D plot visualizing deaths (y-axis) for each surveilance method
        across percent_reduced_infectivity parameter space [0,1] (x-axis).
        """
        V = 20
        lockdown_deaths = np.zeros((V, 3)) # we have three nets to test
        for i in range(V):
            self.set_all_parameters(self.sequencing_params, 
                                self.blood_params, 
                                self.threat_params, 
                                self.astute_params, 
                                self.SIR_params, 
                                [self.lockdown_params[0], i/V])
            for j in range(num_runs):
                lockdown_dict_i = self.simulate_with_lockdown()
                for key in lockdown_dict_i:
                    if key == "bloodnet":
                        lockdown_deaths[i][0] += lockdown_dict_i[key][0][:,:,6][:,0][-1]
                    elif key == "threatnet":
                        lockdown_deaths[i][1] += lockdown_dict_i[key][0][:,:,6][:,0][-1]
                    elif key == "astutenet":
                        lockdown_deaths[i][2] += lockdown_dict_i[key][0][:,:,6][:,0][-1]
        
        lockdown_deaths = np.divide(lockdown_deaths, num_runs)

        plt.figure()
        plt.grid()
        plt.title("Deaths by model with varied reduced_infectivity")
        plt.plot(np.linspace(0, 1, V), lockdown_deaths[:, 0], 'red', label='Bloodnet')
        plt.plot(np.linspace(0, 1, V), lockdown_deaths[:, 1], 'blue', label='Threatnet')
        plt.plot(np.linspace(0, 1, V), lockdown_deaths[:, 2], 'green', label='Astutenet')
        plt.xlabel('% infectivity reduction after detection')
        plt.ylabel('Numbers of deaths')
        plt.ylim([0,np.amax(lockdown_deaths)])
        plt.legend()

        plt.show()

        return


    # METHODS FOR ESTIMATING COST SAVINGS OF SURVEILANCE METHODS

    # TODO: multiplex and average

    def simulate_with_lockdown(self, step_type="stochastic", vaccine_deploy_date=150):
        """Carries out a simulation of the model with the stated parameters,
        creating pop numpy matrices for each community that store information 
        about the simulated population/communities for each time step. Also
        tracks if surveilance methods have detected the pathogen. When a method
        detects a pathogen, the simulation forks: in one simulation, the pathogen
        is "detected" and infectivity is reduced. In the other, the simulation
        carries on as normal.
        
        ARGS:
        step_type: determines whether steps should be calculated deterministically or stochastically.
            - Valid inputs: "stochastic" or "deterministic"
        vaccine_deploy_date: number of days after detection that a vaccine is deployed, reducing the death rate

        RETURNS:
        forked_dict: a dictionary of tuples. Each tuple contains a simulated population
        and the day the pathogen was detected and the simulation was forked. The key of
        each value is the surveilance method that detected the pathogen and thus forked
        the simulation.
        """
        prob_list = [["bloodnet", self.blood_prob], 
                     ["astutenet", self.astute_prob], 
                     ["threatnet", self.threat_prob]]
        forked_dict = {}

        if step_type not in ["stochastic","deterministic"]:
            print("Error: You have input an invalid type for step_type. Please input either 'stochastic' or 'deterministic'.")
            return
        for i in range(1, self.tmax+1):
            if step_type == "stochastic":
                self.pop[i] = self.stochastic_SIRstep(self.pop, self.SIR_params, self.community_params, i)    
            else:
                self.pop[i] = self.deterministic_SIRstep(self.pop, self.SIR_params, self.community_params, i)
            self.bloodnet(i)
            self.bloodnet_community(i)
            self.threatnet(i)
            self.astutenet(i)
            for prob in prob_list:
                if self.day_of_detection(prob[1]) == i: # check if 
                    forked_pop = copy.deepcopy(self.pop)
                    forked_dict[prob[0]] = (self.forked_simulate(forked_pop, vaccine_deploy_date, start=i), self.day_of_detection(prob[1]))
            # print(self.pop[:,:,6][:,0][i])

        return forked_dict
    
    def forked_simulate(self, pop, vaccine_deploy_date, step_type="stochastic", start=1):
        """Runs a forked simulation of the Population's self.pop after a pathogen has
        been detected. This simulation has reduced infectivity from quarantine mandates,
        and after vaccine_deploy_date days will have 99% reduced death rate from deployment
        of vaccines.

        ARGS:
        pop: 3D numpy array. Values for each population bin at the current time step
            - S: susceptible population
            - E: exposed population with the disease who aren't infectious
            - I: infectious population that isn't symptomatic
            - Sy: symptomatic population
            - R: recovered population
            - D: dead population
            - TotI: total population with the disease
            - new_exposed: newly exposed individuals in the population
            - new_infectious: newly infected individuals in the population
        vaccine_deploy_date: number of days after detection that a vaccine is deployed, reducing the death rate
        step_type: determines whether steps should be calculated deterministically or stochastically.
            - Valid inputs: "stochastic" or "deterministic"
        start: the day at which the forked simulation should start

        RETURNS:
        pop: an updated version of the original pop numpy array post-simulation
        """
        if step_type not in ["stochastic","deterministic"]:
            print("Error: You have input an invalid type for step_type. Please input either 'stochastic' or 'deterministic'.")
            return
        
        p_stay_at_home, percent_reduced_infectivity = self.lockdown_params
        forked_SIR_params = list(copy.deepcopy(self.SIR_params))
        forked_SIR_params[0] *= percent_reduced_infectivity

        for i in range(start, self.tmax+1):
            if i == vaccine_deploy_date + start:
                vaccine_SIR_params = list(copy.deepcopy(forked_SIR_params))
                # vaccine_SIR_params[0] *= 0.01 # set infectivity to 0 (not implemented rn because vaccines tend to not reduce infectivity)
                vaccine_SIR_params[5] *= 0.01 # set death rate to 0
                forked_SIR_params = vaccine_SIR_params
            if step_type == "stochastic":
                pop[i] = self.stochastic_SIRstep(pop, forked_SIR_params, self.community_params, i)    
            else:
                pop[i] = self.deterministic_SIRstep(pop, forked_SIR_params, self.community_params, i)

        return pop
    
    def plot_lockdown_simulations(self):
        """Calls simulate_with_lockdown, then plots the control case (no detection)
        against each forked detection simulation. Also prints the following statistics
        for each plot:
            - Model name
            - Day of detection
            - Deaths before detection
            - Infections before detection
            - Total deaths in simulation

        ARGS:
        none.

        RETURNS:
        none.
        """
        forked_dict = self.simulate_with_lockdown()
        for key in forked_dict:
            print(f'Model: {key:<10}, Day of Detection: {int(forked_dict[key][1]):<4}, Deaths before detection: {int(forked_dict[key][0][:,:,6][:,0][forked_dict[key][1]]):<8}, Infections before detection: {int(forked_dict[key][0][:,:,7][:,0][forked_dict[key][1]]):<8} Deaths: {int(forked_dict[key][0][:,:,6][:,0][-1]):<8}')
            self.plot_sim(forked_dict[key][0], f'Detection using {key}')
        print(f'Model: {"no model":<10}, Day of Detection: {"N/A":<4}, Deaths before detection: {"N/A":<8}, Infections before detection: {"N/A":<8} Deaths: {self.pop[:,:,6][:,0][-1]:<8}')
        self.plot_sim(self.pop, "Epidemiological Model, Total Population Over Time")