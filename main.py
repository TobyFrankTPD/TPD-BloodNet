from simulation import Population
import numpy as np

tmax = 500 # total number of time steps
N = 330000000 # population size
initial_infected = 1 # TODO: Allow the infection to start in multiple communities

#community_params: num_communities, initial_community_sizes movement_matrix
initial_community_sizes = [1]
num_communities = len(initial_community_sizes)
movement_matrix = [1]

# Sample movement matrix for num_communities = 5:
# [0.9, 0.1, 0, 0, 0],
# [0, 0.9, 0.1, 0, 0],
# [0, 0, 0.9, 0.1, 0],
# [0, 0, 0, 0.9, 0.1],
# [0.1, 0, 0, 0, 0.9]

community_params = [num_communities, initial_community_sizes, movement_matrix]

#sequencing_params: true_positive_rate, false_positive_rate
#   Note: false_positive_rate includes samples with no pathogen detected as having a pathogen, and samples with a pathogen but not the pathogen that is exponentially growing
sequencing_params = [0.95, 0.1]
#true_positive_rate: proportion of sequences w/pathogen the test detects (also called statistical power)
#false_positive_rate: proportion of sequences w/out pathogen of interest that the test says has a pathogen

#blood_params: p_donation, p_donation_to_bloodnet
p_donation = 16/330/365 #proportion of people who donate blood each day (https://www.umms.org/-/media/files/ummc/community/blood-facts)
blood_params = [p_donation, 1]

#threat_params: background_sick_rate, p_hospitalized, p_hospital_sequenced
#   background_sick_rate: the proportion of people who are sick with a non-exponentially-growing pathogen
#   p_hospitalized: the proportion of symptomatic people who go to a hospital
#   p_hospital_sequenced: the probability of a hospitalized person getting sequenced
p_hospitalized = 0.2
threat_params = [0.001, p_hospitalized, 0.2]

#astute_params: p_hospitalized, p_doctor_detect, command_readiness
#   p_hospitalized: the proportion of symptomatic people who go to a hospital
#   p_doctor_detect: the probability that a doctor reports a symptomatic case as a new pathogen
#   command_readiness: likelihood of a doctor's report being picked up by the system
astute_params = [p_hospitalized, 0.01, 0.1]

#SIR_params: beta, gamma, inf_time, symp_time, N
SIR_params = [0.5, 0.1, 21, 10]



model_population = Population(N, initial_infected, tmax, community_params)
model_population.set_parameters(sequencing_params, blood_params, threat_params, astute_params, SIR_params)

model_population.simulate()

# model_population.plot_net()

model_population.plot_sim()



# List of Model Params:

# pathogen_params: 0 -> infinity
# beta: [0,inf]
# gamma: [0,inf]
# inf_time: [0, inf]
# symp_time: [0, inf]

# all other params: 0 -> 1
# true_positive_rate: [0,1]
# false_positive_rate: [0,1]
# p_donation: [0,1]
# p_donation_to_bloodnet: [0,1]
# background_sick_rate [0,1]
# p_hospitalized: [0,1]
# p_hospital_sequenced: [0,1]
# p_doctor_detect: [0,1]
# command_readiness: [0,1]

# community params: N/A
# num_communitites (n): [0,N]
# initial_community_sizes: n x 1 numpy array
# movement_matrix: n x n numpy array