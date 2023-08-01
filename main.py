from simulation import Population
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

#   MUTABLE PARAMS, USER CAN CHANGE

tmax = 1000 # total number of time steps
N = 332000000 # population size
initial_infected = 1 # TODO: Allow the infection to start in multiple communities
population_bins = ["S", "E", "I", "Sy", "Asy", "R", "D", "TotI", "new_exposed", "new_inf"]
 
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

community_params = (num_communities, initial_community_sizes, movement_matrix)

#lockdown_params: scaled_infectivity
lockdown_params = (0.2)

#detection_params: threshold, time_delay
detection_params = (0.99, 4)

#SIR_params: beta, gamma, inf_time, symp_time, p_asymp, mu
SIR_params = (0.32, 0.125, 4, 2, 0.1, 0.01)

p_hospitalized = 0.08 # hospitalization rate depends upon the severity of the virus. This is calculated for 

#   IMMUTABLE PARAMS, FINE-TUNED FOR ACCURACY

#sequencing_params: true_positive_rate, false_positive_rate
#   Note: false_positive_rate includes samples with no pathogen detected as having a pathogen, and samples with a pathogen but not the pathogen that is exponentially growing
sequencing_params = (0.99, 0.01)
#true_positive_rate: proportion of sequences w/pathogen the test detects (also called statistical power)
#false_positive_rate: proportion of sequences w/out pathogen of interest that the test says has a pathogen

#blood_params: p_donation, p_donation_to_bloodnet
p_donation = 16/330/365 #proportion of people who donate blood each day (https://www.umms.org/-/media/files/ummc/community/blood-facts)
blood_params = (p_donation, 1)

#threat_params: background_sick_rate, p_hospitalized, p_hospital_sequenced
#   background_sick_rate: the proportion of people who are sick with a non-exponentially-growing pathogen
#   p_hospitalized: the proportion of symptomatic people who go to a hospital
#   p_hospital_sequenced: the probability of a hospitalized person getting sequenced
background_sick_rate = 8.1 / 365 / 332
threat_params = (background_sick_rate, p_hospitalized, 0.2)

#astute_params: p_hospitalized, p_doctor_detect, command_readiness
#   p_hospitalized: the proportion of symptomatic people who go to a hospital
#   p_doctor_detect: the probability that a doctor reports a symptomatic case as a new pathogen
#   command_readiness: likelihood of a doctor's report leading to a substantial pandemic response
astute_params = (p_hospitalized, 0.2, 0.05)



# EXAMPLE CODE FOR RUNNING THE SIMULATION:


model_population = Population(N, initial_infected, tmax, population_bins, community_params)
model_population.set_all_params(sequencing_params, blood_params, threat_params, astute_params, SIR_params, lockdown_params, detection_params)

model_population.simulate()

model_population.plot_sim()

# model_population.plot_net()

# for i in range(5):
#     model_population.plot_lockdown_simulations()

# print(model_population.test_nets(10))



# t_run = 200
# average_one_day = np.zeros((t_run,1))
# num_runs = 50

# for i in range(num_runs):
#     print(i)
#     one_day_i = model_population.one_day_market_shaping_costs()
#     average_one_day = np.add(average_one_day, one_day_i)

# average_one_day = np.divide(average_one_day, num_runs)
# average_one_day = np.round(average_one_day, 2)

# np.set_printoptions(threshold=np.inf)
# for i in range(len(average_one_day)):
#     print(f'Day {i}: {average_one_day[i]}')

# infectious = np.divide(model_population.pop[:,:,2][:,0][0:200], np.amax(model_population.pop[:,:,2][:,0][0:200])/np.amax(average_one_day))
# dead = np.divide(model_population.pop[:,:,6][:,0][0:200], np.amax(model_population.pop[:,:,6][:,0][0:200])/np.amax(average_one_day))

# plt.plot(range(t_run), np.multiply(average_one_day, 1), 'orange', label='Cost Estimate $B')
# plt.plot(range(t_run), infectious, 'red', label='Infectious population')
# plt.plot(range(t_run), dead, 'black', label='Dead population')
# plt.legend()
# plt.xlabel("Day of Detection")
# plt.ylabel("Cost Estimate ($B) of One Day Later")
# plt.show()



# model_population.sequencing_param_tester(3)

# model_population.SIR_param_tester()

# model_population.lockdowm_param_tester()







# LIST OF PARAMETERS:

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