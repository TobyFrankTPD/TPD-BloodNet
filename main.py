from simulation import simulate, plot_sim, plot_net
import numpy as np

tmax = 200 # total number of time steps
N = 9000000 # population size
#params: beta, gamma, inf_time, symp_time, N
params = [1, 0.1, 7, 7, N]
#pop: S, E, I, Sy, R, TotI, new_acquired, new_infectious (TotI = A + I + Sy)
pop = np.zeros((tmax+1,8))
pop[0,:] = [N-1, 0, 1, 0, 0, 1, 0, 0]
t = np.linspace(0, tmax, tmax+1)

#sequencing_params: true_positive_rate, false_positive_rate, sequencing_confidence_threshold
sequencing_params = [0.9, 0.1, 0.99]

#true_positive_rate: proportion of sequences w/pathogen the test detects (also called statistical power)
#false_positive_rate: proportion of sequences w/no pathogen the test says has pathogen
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

p_donation = 16/300/365 #proportion of people who donate blood each day (https://www.umms.org/-/media/files/ummc/community/blood-facts)
#blood_params: detection_thresh_blood, num_daily_donations, p_donation_to_bloodnet
blood_params = [3, p_donation*N, 1]

#threat_params: detection_thresh_threat, p_inf_sequenced
threat_params = [3, 0.04]

pop = simulate(pop, params, tmax)

# plot_sim(pop)

plot_net(pop, blood_params, threat_params, sequencing_params)