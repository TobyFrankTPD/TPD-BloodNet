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

#blood_params: detection_thresh_blood, p_inf_donation, p_donation_to_bloodnet
blood_params = [3, 0.01, 1]

#threat_params: detection_thresh_threat, p_inf_sequenced
threat_params = [3, 0.025]

pop = simulate(pop, params, tmax)

# plot_sim(pop)

plot_net(pop, blood_params, threat_params)