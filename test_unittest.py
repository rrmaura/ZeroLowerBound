import unittest   # The test framework
import numpy as np 
import torch 
from simulation import PolicyNet

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
if __name__ == '__main__':
    unittest.main()