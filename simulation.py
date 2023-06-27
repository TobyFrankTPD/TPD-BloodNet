"""Epidemiological model developed for TPD. Simulates an epidemic using an extended
SIR model with susceptible, exposed, infectious, symptomatic, and recovered buckets.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom
import math

# TODO:
#   - Make a population Class that stores the info instead of numpy arrays
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
        - R: recovered population
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
    """
    def __init__(self, N = 100000, initial_infected = 1, tmax = 200, community_params = [1, [1], [1]]):
        self.N = N
        self.tmax = tmax
        self.community_params = community_params
        num_communities, initial_community_sizes, movement_matrix = self.community_params

        self.pop = np.zeros((self.tmax+1, num_communities+1, 8))
        self.pop[0][0] = [N-1, 0, 1, 0, 0, 1, 0, 0]
        for i in range(num_communities):
            infected_i = initial_infected if i == 0 else 0
            N_i = round(N*initial_community_sizes[i])
            pop_i = [N_i-infected_i, 0, infected_i, 0, 0, infected_i, 0, 0]
            self.pop[0][i+1] = pop_i
        
        self.t = np.linspace(0, tmax, tmax+1)

    def set_parameters(self, sequencing_params, blood_params, threat_params, astute_params, SIR_params):
        self.sequencing_params = sequencing_params
        self.blood_params = blood_params
        self.threat_params = threat_params
        self.astute_params = astute_params
        self.SIR_params = SIR_params

    def set_threshold(self, threshold):
        self.threshold = threshold

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
                self.pop[i] = self.stochastic_SIRstep(self.pop, self.SIR_params, self.community_params, i)
            else:
                self.pop[i] = self.deterministic_SIRstep(self.pop, self.SIR_params, self.community_params, i)

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
        beta, gamma, inf_time, symp_time = params
        S, E, I, Sy, R, TotI = pop[t-1][0:6]

        # calculate changes in population bins
        new_exposed = (S*(I+Sy)*beta)/self.N
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
        beta, gamma, inf_time, symp_time = params
        num_communities, initial_community_sizes, movement_matrix = community_params
        pop_t = np.zeros((num_communities+1, 8))

        for i in range(num_communities):
            S, E, I, Sy, R, TotI = pop[t-1][i+1][0:6]
            N_i = sum([S, E, I, Sy, R])

            # calculate changes in population bins
            p_exposed = (beta*(I+Sy)*S)/(N_i**2)
            p_recovered = (Sy*gamma)/N_i
            new_exposed = np.random.binomial(S, p_exposed)
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
            S1 = S-new_exposed
            E1 = E+new_exposed-new_infectious
            I1 = I+new_infectious-new_symptomatic
            Sy1 = Sy+new_symptomatic-new_recovered
            R1 = R+new_recovered
            TotI1 = E1 + I1 + Sy1
            pop_t[i+1] = [S1, E1, I1, Sy1, R1, TotI1, new_exposed, new_infectious]
            pop_t[0] = [sum(x) for x in zip(pop_t[0], [S1, E1, I1, Sy1, R1, TotI1, new_exposed, new_infectious])]

        pop_t_moved = np.dot(movement_matrix, pop_t[1:])

        output = np.vstack((pop_t[0], pop_t_moved))

        return output

    def plot_sim(self):
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
            - R
        """
        plt.figure()
        plt.grid()
        plt.title("Epidemiological Model")
        plt.plot(self.t, self.pop[:,:,0][:,0], 'orange', label='Susceptible')
        plt.plot(self.t, self.pop[:,:,1][:,0], 'blue', label='Exposed')
        plt.plot(self.t, self.pop[:,:,2][:,0], 'r', label='Infectious')
        plt.plot(self.t, self.pop[:,:,3][:,0], 'g', label='Symptomatic')
        plt.plot(self.t, self.pop[:,:,4][:,0], 'yellow', label='Recovered')
        plt.xlabel('Time t, [days]')
        plt.ylabel('Numbers of individuals')
        plt.ylim([0,self.N])
        plt.legend()

        plt.show()

    # METHODS FOR RUNNING AND VISUALIZING VARIOUS SURVEILANCE METHODS' EFFICIENCIES

    def bloodnet(self, community_i=0):
        """Given a pop numpy matrix, calculates the probability of a bloodnet
        surveilance approach detecting the pathogen within the population.

        BloodNet: every day, a certain proportion of people donate blood, which are then sequenced and
        tested. Positive tests (indicating presence of a pathogen) occur at some rate even when there 
        is no pathogen. When positive tests increase, the chance of a true positive goes up. At some
        point, the chance of a true positive is high enough to reject the null hypothesis of no true 
        positives.
        
        ARGS:
        community_i: instructions for which community to use for calculations. Default is total population (0).

        RETURNS:
        prob_detect: a [tmax+1,1] dimentional vector. For each time step,
        stores the probability of a bloodnet model detecting the pathogen.
        """
        temp_pop = self.pop[:,0:,][:,community_i] # isolates the desired population data from the population tensor
        p_donation, p_donation_to_bloodnet = self.blood_params
        true_positive_rate, false_positive_rate = self.sequencing_params
        prob_detect = np.zeros(self.tmax+1)

        for i in range(1,self.tmax+1):
            don_pop = self.N-temp_pop[i][3]
            num_samples = np.random.binomial(don_pop, p_donation * p_donation_to_bloodnet)

            if num_samples == 0:
                prob_detect[i] = 0.5
            else:
                x = np.arange(0, num_samples+1)

                # TODO: the test below assumes that the null hypothesis is 0 true positive tests. 
                #   In reality, there are true positives without an exponentially growing pathogen
                p_x_positives_null = binom.sf(x, num_samples, false_positive_rate) #probability distribution of obsering at least x positive results given no true positives

                p_infected = (temp_pop[i][1]+temp_pop[i][2])/don_pop #probability one person is infected
                p_clean = 1 - p_infected
                p_positive = true_positive_rate*p_infected + false_positive_rate*p_clean #probability a sequencing test will return a positive result
                num_positive_tests = np.random.binomial(num_samples, p_positive)

                # print(f'community: {community_i}, day: {i:<3}, don_pop: {round(don_pop):<7}, num_samples: {round(num_samples,6):<12}, num_positives: {round(num_positive_tests,3):<12}, E+I: {round(pop[i][1]+pop[i][2]):<12}, Sy: {round(pop[i][3]):<12}, prob_detect: {round(1-p_x_positives_null[round(num_positive_tests)],6):<7}')

                prob_detect[i] = 1-p_x_positives_null[num_positive_tests]

        return prob_detect

    def bloodnet_community(self):
        """Calls the bloodnet detection model on each sub-community
        of the population.
        
        ARGS:
        none.

        RETURNS:
        prob_detects: a [num_communities, tmax+1] dimentional vector. For each time step,
        stores the probability of a bloodnet model detecting the pathogen for each community.
        """
        num_communities, initial_community_sizes, movement_matrix = self.community_params
        prob_detects = np.zeros((num_communities, self.tmax+1))

        for i in range(num_communities):
            prob_detects[i] = self.bloodnet(i+1)

        return prob_detects


    def threatnet(self, community_i=0):
        """Given a pop numpy matrix, calculates the probability of a threatnet
        surveilance approach detecting the pathogen within the population.

        ThreatNet: symptomatic people and people sick with other illnesses visit hospitals with 
        a fixed probability, some of whom will get sequenced. Positive tests (indicating presence 
        of a pathogen) occur at some rate even when there is no pathogen. When positive tests 
        increase, the chance of a true positive goes up. At some point, the chance of a true 
        positive is high enough to reject the null hypothesis of no true positives.
        
        ARGS:
        none.

        RETURNS:
        prob_detect: a [tmax+1,1] dimentional vector. For each time step,
        stores the probability of a threatnet model detecting the pathogen.
        """
        temp_pop = self.pop[:,0:,][:,community_i] # isolates the desired population data from the population tensor
        background_sick_rate, p_hospitalized, p_hospital_sequenced = self.threat_params
        true_positive_rate, false_positive_rate = self.sequencing_params

        Sy = temp_pop[:,3]
        prob_detect = np.zeros(self.tmax+1)

        for i in range(1, self.tmax+1):
            num_sick = (self.N - Sy[i]) * background_sick_rate + Sy[i]
            num_sequenced = np.random.binomial(num_sick, p_hospitalized*p_hospital_sequenced)

            # If 10% of the population is sick, assume the pathogen has been detected.
            # Added this in to speed up the computation, it is definitely an assumption though
            x = np.arange(0, num_sequenced+1)

            p_x_positives_null = binom.sf(x, num_sequenced, false_positive_rate)

            p_infected = (Sy[i])/num_sick #probability one person is infected out of the group of sequenced people
            p_clean = 1 - p_infected
            p_positive = true_positive_rate*p_infected + false_positive_rate*p_clean #probability a sequencing test will return a positive result
            num_positive_tests = np.random.binomial(num_sequenced, p_positive)

            prob_detect[i] = 1-p_x_positives_null[num_positive_tests]

        return prob_detect

    def astutenet(self, community_i=0):
        """Given a pop numpy matrix, calculates the probability of an astute doctor
        successfully discovering and reporting a novel pathogen.

        When people get sick, they sometimes go to the hospital. These people are
        seen by doctors, and if enough people have novel symptoms then an astute
        doctor might realize there is something going on. If the doctor raises
        the alarm and their alarm is heard, then the new pathogen is detected.

        ARGS:
        community_i: instructions for which community to use for calculations. Default is total population

        RETURNS:
        prob_detect: a [tmax+1,1] dimentional vector. For each time step,
        stores the probability of an astute doctor successfully reporting a novel pathogen.
        """
        p_hospitalized, p_doctor_detect, chi = self.astute_params
        temp_pop = self.pop[:,0:,][:,community_i] # isolates the desired population data from the population tensor

        Sy = temp_pop[:,3]
        prob_detect = np.zeros(self.tmax+1)

        for i in range(1, self.tmax+1):
            num_symp_hospitalized = np.random.binomial(Sy[i], p_hospitalized)
            num_hospital_reports = np.random.binomial(num_symp_hospitalized, p_doctor_detect)

            # not stochastic, can make it stochastic w/a cdf. This function is very arbitrary lol
            p_investigation = 1 / (1 + (50/chi) * math.e**(-2 * chi * num_hospital_reports))
            prob_detect[i] = p_investigation
        
        return prob_detect

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

        blood_prob = self.bloodnet()
        community_blood_probs = self.bloodnet_community()
        astute_prob = self.astutenet()
        threat_prob = self.threatnet()
        
        temp_pop = self.pop[:,0:,][:,0] # isolates the aggregate population data from the population tensor
        p_inf = np.zeros(nrows) # p_inf: percent of people who are not susceptible (E, I, Sy, or R)
        p_symp = np.zeros(nrows)
        for i in range(nrows):
            p_inf[i] = (temp_pop[i][1]+temp_pop[i][2])/sum(temp_pop[0][0:5])
            p_symp[i] = (temp_pop[i][3])/sum(temp_pop[0][0:5])

        # Astute Doctor Total Model
        plt.figure()
        plt.title(f'Astute Doctor Total Population Model')
        plt.plot(self.t, astute_prob, 'red', label='Prob of Detect (AstuteNet)')
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
        plt.plot(self.t, threat_prob, 'green', label='Prob of Detect (ThreatNet)')
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
        plt.plot(self.t, blood_prob, 'red', label='Prob of Detect (BloodNet)')
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
            plt.plot(self.t, community_blood_probs[i], color, label=f'Prob of Detect (BloodNet C{i+1})')
            plt.plot(self.t, p_inf, 'blue', label='% infected/recovered')
            plt.plot(self.t, p_symp, 'yellow', label='% symptomatic')
            plt.xlabel('Time t, [days]')
            plt.ylabel('Numbers of individuals')
            plt.ylim([0,1])
            plt.legend()
            plt.show()  

    # METHODS FOR RUNNING AND VISUALIZING PARAMETER SEARCHING

    def day_of_detection(self, prob_detect, threshold):
        """Given the output of a surveilance method (prob_detect), calculate
        the day when the method detects the pathogen. This is defined as the
        day when the probability of detection exceeds the threshold for seven conscutive
        days.

        ARGS:
        prob_detect: a tmax x 1 numpy array storing the probability the method detects a pathogen for a given day
        threshold: probability threshold that must be exceeded for seven consecutive days
        
        RETURNS:
        day_of_detection: the day the method detects the pathogen, as defined above.
        """
        def threshold_checker(x):
            return 1 if x > threshold else 0

        v_thresh = np.vectorize(threshold_checker)
        prob_detect = v_thresh(prob_detect)

        for i in range(len(prob_detect)):
            if 0 not in prob_detect[i:i+7]:
                return i
        
        return self.tmax + 1


    
    def test_nets(self):
        """Given a simulated population, calculates the day
        when each surveilance system detects the pathogen.

        ARGS:
        threshold: probability threshold that must be exceeded for seven consecutive days

        RETURNS:
        None.
        """
        blood_prob = self.bloodnet()
        # community_blood_probs = self.bloodnet_community()
        astute_prob = self.astutenet()
        threat_prob = self.threatnet()

        blood_prob_thresh = self.day_of_detection(blood_prob, self.threshold)
        astute_prob_thresh = self.day_of_detection(astute_prob, self.threshold)
        threat_prob_thresh = self.day_of_detection(threat_prob, self.threshold)

        # 0 = bloodnet, 1 = astutenet, 2 = threatnet
        thresh_dict = {blood_prob_thresh: "B", astute_prob_thresh: "A", threat_prob_thresh: "T"}
        best_net = min(blood_prob_thresh, astute_prob_thresh, threat_prob_thresh)

        return [thresh_dict[best_net], best_net]
    
    def sequencing_param_tester(self):
        best_net_list = np.chararray((10, 10), unicode=False)
        best_net_values = np.zeros((10, 10))
        for i in range(0, 10):
            for j in range(0, 10):
                print(i, j)
                self.set_parameters([i/10, j/10], self.blood_params, self.threat_params, self.astute_params, self.SIR_params)
                best_net_list[i][j], best_net_values[i][j] = self.test_nets()
        
        self.sequencing_heatmap = sns.heatmap(best_net_values, linewidth=0.5, annot=best_net_list, fmt="")
        plt.title("Heatmap of Detection Methods with Varied Sequencing_Param")
        plt.xlabel('False_positive_rate')
        plt.ylabel('False_negative_rate')
        plt.show()