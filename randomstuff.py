import torch 
from definitions import *
N_banks = 150

E = torch.ones(100, N_banks)
percent_assets_to_reserves = initialize_and_load_NN()
a = t.load('percent_assets_to_reserves.pt')
percent = percent_assets_to_reserves(E)
torch.isnan(percent).any()
print("hello world")


# Why are the banks not buying more Loans? that provides bigger profits...
# yet they prefet to buy more reserves...