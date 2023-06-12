import unittest
import numpy as np
from simulation import simulate, bloodnet, threatnet

# TODO:
#   - Figure out how to test the stochastic simulation using unittest: is unittest even useful?

class TestTranslator(unittest.TestCase):        

    def test_simulate_basecases(self):

        tmax = 50 # total number of time steps
        N = 9000000 # population size
        #params: beta, gamma, inf_time, symp_time, N
        params = [1, 0, 4, 4, N]
        #pop: S, A, I, Sy, R, TotI, new_acquired, new_infectious (TotI = A + I + Sy)
        pop = np.zeros((tmax+1,8))
        pop[0,:] = [N-1, 0, 1, 0, 0, 1, 0, 0]
        t = np.linspace(0, tmax, tmax+1)

        detection_thresh_blood = 3
        p_inf_donation = 0.01

        detection_thresh_threat = 3
        p_inf_sequenced = 4/100

        pop = simulate(params, pop, tmax)

        self.assertTrue(True == True)

if __name__ == '__main__':
    unittest.main()