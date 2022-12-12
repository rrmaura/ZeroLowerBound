import unittest   # The test framework
import numpy as np 
import torch 
from simulation import *


N_BANKS = 100
class TestSimulation(unittest.TestCase):
    def test_network_not_nan(self):
        # test that the network is not nan
        E = torch.ones(100,N_BANKS)
        loans_net = PolicyNet(input_size=N_BANKS, 
                      hidden_size=10, 
                      output_size=N_BANKS)

        L = loans_net(E)
        self.assertFalse(torch.isnan(L).any())
        return 

    def interest_rates_reasonable(self):
        # test that the interest rates are reasonable
        Ei = torch.rand(N_banks)
        Li = loans_net(Ei)
        Di = Ei*lmda/(1-lmda)
        Mi = Di + Ei - Li
        profits = Mi*rM + Li*rL(Li) - Di*rD(Di) - cost(Li,Di)
        dividends = propor_div*profits

    def FOCs(self): 
        # test that the FOCs are satisfied

        
        # load solution neural network
        loans_net = PolicyNet(input_size=N_BANKS,
                                hidden_size=10,
                                output_size=N_BANKS)
        loans_net.load_state_dict(torch.load('loans_net.pt'))

        Ei = torch.rand(N_banks)
        Li = loans_net(Ei)
        Di = Ei*lmda/(1-lmda)
        Mi = Di + Ei - Li
        

if __name__ == '__main__':
    unittest.main()