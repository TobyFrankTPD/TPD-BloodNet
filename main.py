from simulation import simulate, plot_sim, plot_net
import numpy as np

#sequencing_params: true_positive_rate, false_positive_rate
#   Note: false_positive_rate includes samples with no pathogen detected as having a pathogen, and samples with a pathogen but not the pathogen that is exponentially growing
sequencing_params = [0.95, 0.1]

#true_positive_rate: proportion of sequences w/pathogen the test detects (also called statistical power)
#false_positive_rate: proportion of sequences w/out pathogen of interest that the test says has a pathogen
#p_infected_positive: the probability that a positive test indicates an infected individual
#num_trials_for_detection: # of positive tests required for sequencing_confidence_threshold chance of at least one true positive
#sequencing_confidence_threshold: probability threshold at which the sequencing method "detects" the pathogen (ie: countermeasures begin)

#if I choose one person from a population and they get a positive on the test, then the chance that they have the pathogen is calculated below
#p(infected) = TotI/N
#p(positive | infected) = 0.9 - true_positive_rate
#p(clean) = (1-TotI)/N
#p(positive | clean) = 0.1 - false_positive_rate
#p(positive) = p(positive | clean)*p(clean) + p(positive | infected)*p(infected)
###p(infected | positive) = (p(positive | infected) * p(infected)) / (p(positive))

p_donation = 16/330/365 #proportion of people who donate blood each day (https://www.umms.org/-/media/files/ummc/community/blood-facts)
#blood_params: detection_thresh_blood, num_daily_donations, p_donation_to_bloodnet
blood_params = [p_donation, 1]

initial_community_sizes = [0.5, 0.5]
num_communities = len(initial_community_sizes)
movement_matrix = [[0.9, 0.1],[0.1, 0.9]]

community_params = [num_communities, movement_matrix]


#threat_params: background_sick_rate, p_sick_sequenced
#   background_sick_rate: the proportion of people who are sick with a non-exponentially-growing pathogen
#   p_sick_sequenced: the probability of one sick person (w/out pathogen of interest) getting sequenced at a hospital
threat_params = [0.001, 0.04]

tmax = 200 # total number of time steps
N = 1000000 # population size
initial_infected = 1 # TODO: Allow the infection to start in multiple communities
#params: beta, gamma, inf_time, symp_time, N
params = [0.5, 0.1, 7, 7, N]

#pop: S, E, I, Sy, R, TotI, new_acquired, new_infectious (TotI = A + I + Sy)
pop = np.zeros((tmax+1, num_communities+1, 8))
pop[0][0] = [N-1, 0, 1, 0, 0, 1, 0, 0]
for i in range(num_communities):
    infected_i = initial_infected if i == 0 else 0
    N_i = round(N*initial_community_sizes[i])
    pop_i = [N_i-infected_i, 0, infected_i, 0, 0, infected_i, 0, 0]
    pop[0][i+1] = pop_i

t = np.linspace(0, tmax, tmax+1)

pop = simulate(pop, params, community_params, tmax)

# plot_sim(pop)

plot_net(pop, blood_params, threat_params, sequencing_params)