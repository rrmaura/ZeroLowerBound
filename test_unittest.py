import unittest   # The test framework
import numpy as np 
import torch 
from definitions import *

class TestSimulation(unittest.TestCase):
    def test_network_not_nan(self):
        # test that the network is not nan
        E = torch.ones(100,N_banks)
        percent_assets_to_reserves = initialize_and_load_NN()

        percent = percent_assets_to_reserves(E)
        self.assertFalse(torch.isnan(percent).any())
        return 

    def test_interest_rates_reasonable(self):
        size = torch.ones(N_banks) # initial size of all banks #TODO: TRAIN WITH DIFFERENT SIZES
        percent_assets_to_reserves = initialize_and_load_NN()
        # test that the interest rates are reasonable
        Ei = torch.rand(N_banks)
        total_assets = t.add(Ei, Ei*lmda/(1-lmda))
        Mi = t.mul(percent_assets_to_reserves(Ei), total_assets)
        Di = Ei*lmda/(1-lmda)
        Li = Di + Ei - Mi
        profits = Mi*rM + Li*rL(Li) - Di*rD(Di) - cost(Li,Di, size)
        dividends = propor_div*profits

    def test_FOCs(self): 
        # test that the FOCs are satisfied
        size = torch.ones(N_banks) # initial size of all banks #TODO: TRAIN WITH DIFFERENT SIZES

        # load solution neural network
        percent_assets_to_reserves = initialize_and_load_NN()
        Ei = torch.rand(N_banks)
        total_assets = t.add(Ei, Ei*lmda/(1-lmda))
        Mi = t.mul(percent_assets_to_reserves(Ei), total_assets)
        Di = Ei*lmda/(1-lmda)
        Li = Di + Ei - Mi
        
        # the following conditions are on the version 13 of the paper
        # equation 97

        # propor_div + beta*(1-propor_div)*dV/dEi = 0
        
        def deterministic_value_function(Ei, size): 
            value = t.zeros_like(Ei)
            discount = 1 #SDF
            for time in range(MAX_TIME):
                Ei, size, dividends = next_equity_size_and_dividents(Ei, 
                                                                    size, 
                                                                    time, 
                                                                    percent_assets_to_reserves) # update Ei
                # value += discount*dividends 
                value = t.add(value, t.mul(discount, dividends))
                discount *= discount*beta# TODO: add stochastic discount factor
            return value
        Delta = 0.0001
        V = deterministic_value_function(Ei, size) # only i-th bank 
        for i in range(N_banks):
            V_i = V[i]
            Ei_delta = Ei.clone()
            Ei_delta[i] += Delta # only the first bank changes
            

            profits = Mi*rM + Li*rL(Li) - Di*rD(Di) - cost(Li,Di, size)
            Li_delta = Li.clone()
            Li_delta[i] += Delta
            profits_deltaL = Mi*rM + Li_delta*rL(Li_delta) - Di*rD(Di) - cost(Li_delta,Di, size)
            dprofits_dLi = (profits_deltaL - profits)/Delta
            print(round(dprofits_dLi[0].item(),3))
            assert abs(dprofits_dLi[0].item()) < 0.01

            # now the same with respect to Di
            Di_delta = Di.clone()
            Di_delta[i] += Delta
            profits_deltaD = Mi*rM + Li*rL(Li) - Di_delta*rD(Di_delta) - cost(Li,Di_delta, size)
            dprofits_dDi = (profits_deltaD - profits)/Delta
            print(round(dprofits_dDi[0].item(),3))
            assert abs(dprofits_dDi[0].item()) < 0.01

            # equation 83
            # Di - lmda * Li = lmda * Mi
            # print(round(Di[i].item() - lmda*Li[i].item() - lmda*Mi[i].item(),3))
            assert abs(Di[i].item() - lmda*Li[i].item() - lmda*Mi[i].item()) < 0.01

# test FOCs
# test that the FOCs are satisfied
test = TestSimulation()
test.test_FOCs()

if __name__ == '__main__':
    unittest.main()