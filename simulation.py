# TODO peace of mind. Ensure that without any shocks,
#  the system is stable and the solution is the analytical one
# TODO: add shocks to the system
# TODO: add f(n) function cost as a function of size of firm n
# TODO: r = 1+R ? or R = 1+r ?

"""
this program is a simulation of a macroeconomic model. This
is a partial equilibrium model, with only heteorgeneous banks. 
Banks can choose Loans (L), Deposits (D), Reserves (M) and Equity (E)
such that M+L = D+E
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# TODO: calibration of parameters. We want realistic values 
# Parameters:
n_epochs = 100 # number of epochs
N_banks = 100 # number of banks
T = 30 # number of time steps
beta = 0.9 # discount factor
# TODO: use utility function of households and stochastic discount factor
lmda = 0.5 # legal ratio of deposits to assets
propor_div = 0.1 # proportion paid out as dividends
rM = 1.03 # interest rate central bank

# we assume linear demand of Lonad and Deposit
phi_L_0 = torch.tensor(1) # intercept
phi_L_1 = torch.tensor(-0.01) # slope (negative)
phi_D_0 = torch.tensor(1)
phi_D_1 = torch.tensor(+0.01)

# we assume the cost is linear in L and D
cost_L = 1
cost_D = 1

# define functions of demand (depends on total)
def rL(Li):
    total_L = torch.sum(Li)
    return torch.add(phi_L_1*total_L, phi_L_0)

def rD(Di):
    total_D = torch.sum(Di)
    # return torch.sum(phi_D_0, phi_D_1*total_D)
    # zero lower bound: the rate for Deposits cannot be lower than 0 
    return torch.max(torch.add(phi_D_1*total_D, phi_D_0), torch.tensor(1.0))

def cost(Li,Di):
    # TODO: am I breaking the graph? Do I need torch.multiply(const_L, L) ?
    # TODO: add f(n) function cost as a function of size of firm n
    return torch.add(cost_L*Li, cost_D*Di) 

# try method 1 from Deep Learning for solving dynamic economic models.
# by Lilia Maliar, Serguei Maliar, Pablo Winant 

# E is exogenous
# we need 1 neural network to approximate the loans L
# D gets determined by credit regulations D/(M+L) <= lmda
# the last equation is equivalent to D <= E lmda/(1-lmda)
# the reserves M will be determined by accounting M+L = D+E 

# the dividends are mu*(M*rM + L*rL - D*rD - Cost)
# equity next period is  (1-mu)*(M*rM + L*rL - D*rD - Cost)

class PolicyNet(nn.Module):
    # 2 hidden layer neural network
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

loans_net = PolicyNet(input_size=N_banks, 
                      hidden_size=10, 
                      output_size=N_banks)

def equity_next_and_dividends_f(Ei):
    Di = Ei*lmda/(1-lmda)
    Li = loans_net(Ei)
    Mi = Di + Ei - Li
    return (1-propor_div)*(Mi*rM + Li*rL(Li) - Di*rD(Di) - cost(Li,Di)), \
            propor_div*(Mi*rM + Li*rL(Li) - Di*rD(Di) - cost(Li,Di))

def objective():
    Ei = torch.rand(N_banks)
    value = torch.zeros_like(Ei)
    discount = 1 
    for _ in range(T):
        Ei, dividends = equity_next_and_dividends_f(Ei) # update Ei
        value += discount*dividends 
        discount *= discount*beta# TODO: stochastic discount factor
    return torch.sum(value)

optimizer = optim.Adam(loans_net.parameters(), lr=0.001)

for _ in tqdm(range(n_epochs)):
    # keep track of the training error
    losses = []
    optimizer.zero_grad()
    loss = -objective() # we want to maximize the objective
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# plot the training error
plt.plot(losses)
plt.show()


# # plot the policy function for the first bank from E = 0 to E = 1
# # assume all other banks have the same equity = 1
# E = torch.ones(100,N_banks)
# E[:,0] = torch.linspace(0,1,100)
# L = loans_net(E)
# equity_first_bank = E[:,0].detach().numpy()
# loan_first_bank = L[:,0].detach().numpy()
# plt.plot(equity_first_bank, loan_first_bank)
# plt.show()

# # what is the problem with the previous code?


