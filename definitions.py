"""
this program is a simulation of a macroeconomic model. This
is a partial equilibrium model, with only heteorgeneous banks. 
Banks can choose Loans (L), Deposits (D), Reserves (M) and Equity (E)
such that M+L = D+E
"""
# TODO: parameters - calibration?
# is INITIAL_EQUITY = 1 reasonable? How many banks? this affectgs the amount of loans
# think about the rL function and why is it so low
# think about why the reserves are so low (close to 0%)
# stochastic discount factor
# Y mana for households in demand of deposits functino rD
# sigma, the variance of the shock of cost (we have to think about units)
# cost_L and cost_D feel too low? 
# hidden_size bigger ( is 10 small?). try 64, 128? I dont think more layers are needed
# TODO: think 
# is it reasonable that the proportion of reserves increases linearly with E?
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# seeds
t.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Parameters:
MAX_EQUITY = 100 # max equity used for training the NN
INITIAL_EQUITY = 1 # initial equity of the banks
N_banks = 150 # number of banks (TODO: change it to 4236)
T = 150 # number of time steps
theta = 0.5 # risk_aversion , inverse of the elasticity of substitution
beta = 0.96 # discount factor
# TODO: add stochastic discount factor (utility function of households)
SDF = 0.97 * beta #Stochastic discount factor 
lmda = 0.85 # legal ratio of deposits to assets (0.97 in calibration, but this
# value gives a more reasonable ratio of deposits to equity )
propor_div = 0.04 # proportion paid out as dividends (0.02<mu<0.05)
rM = 0.03 # interest rate central bank
sigma = 0.01 # variance of the shock of cost (we have to think about units)
R_M = 1 + rM 


# we assume the cost is linear in L and D
cost_L = 0.0001
cost_D = 0.0001

# define functions of demand (depends on total L and D)

    
def rL(Li):
    total_L = t.sum(Li)
    K = total_L 
    K = t.max(K, t.tensor(0.00001)) # stop division by 0
    # TODO: in case K is 0, give a warning
    
    alpha = 0.3
    # The formula of returns for the firm implies:
    # r = alpha * k**(alpha-1)
    r_L = t.mul(alpha, t.pow(K, alpha-1)) 
    return r_L

def R_L(Li):
    # The demand for loans is not linear in the total amount of loans
    # it should come determined by the demand for deposits
    r_L = rL(Li)
    R_L = t.add(1, r_L) # R = 1+r
    return R_L

def rD(Di):
    # The demand for deposits is not linear in the total amount of deposits
    # it is derived from the household maximization problem

    # TODO: what is Y in the household problem?
    # we know that the ration Y/D is around 10% - 20%

    Y = 100
    total_D = t.sum(Di)
    # The formula of returns for the firm implies:
    # r**((theta-1)/(theta)) = beta**(1/theta) * (Y-D)/D 
    # or 
    # r = (  beta**(1/theta) * (Y-D)/D  )**(theta/(theta-1))
    # or 
    # r = c * base**expon 
    # where 
    # c = beta**1/(theta-1),  expon=(theta/(theta-1)), base = (Y-D)/D

    exponent = theta/(theta-1)
    constant =  beta**(1/(theta-1))
    base = t.add(t.div(Y, total_D),-1)

    rD = t.mul(constant, t.pow(base, exponent))

    # zero lower bound
    rD = t.max(rD, t.tensor(0.0))
    return rD
   

def R_D(Di):
    r_D = rD(Di)
    R_D = t.add(1, r_D) # R = 1+r
    return R_D


def size_multiplier_f(size):
    """
    This function returns the multiplier of the cost of the bank. 
    That is, the cost of the bank is proportional to the size of the bank.
    In particular, COST = f(size_i) * (cost_L * Li  + cost_D * Di). 

    arguments:
    size: tensor of size N_banks, with the size of each bank

    returns:
    multiplier: tensor of size N_banks, with the multiplier of the cost f(size)
    """
    # properties of the function:
    # 1) decreasing function of size (the bigger the bank, the lest cost per loan)
    # 2) the function is 1 for size=1
    # This ensures that, in the limit of competitive equilibrium, when all banks
    # are of the same size, and size = 1, then... TODO: check this
    return t.divide(t.tensor(1.0), size)

def size_of_bank_after_mergers(size, Ei):
    # TODO: does this create concentration? or should I give all the size to the
    # bank with highest equity?

    """
    takes the size of all banks and the equity of all banks
    and returns the new size of all banks.
    
    To do this, it checks which banks are bankrupt, and then
    it splits the size of the bankrupt banks among the other banks.
    the probability of a bank A buying/merging with another bankrupt bank B is
    proportional to the equity of the bank A.
    
    parameters:
    size: tensor of size N_banks, with the size of each bank
    Ei: tensor of size N_banks, with the equity of each bank

    returns:
    size: tensor of size N_banks, with the new size of each bank
    """
    # size of bankrupt banks is the sum of the size of all banks where Ei = 0
    bankrupt_mask = t.eq(Ei, t.tensor(0.0))
    size_bankrupt = t.sum(t.masked_select(size, bankrupt_mask))

    # probability of buying a bankrupt bank is proportional to the equity of the bank
    prob_buy_bankrupt = t.div(Ei, t.sum(Ei))
    
    new_size = t.add(size, t.mul(size_bankrupt, prob_buy_bankrupt))
    return new_size 

def cost(Li,Di, size):
    size_multiplier = size_multiplier_f(size) 
    cost_without_noise = t.add(t.mul(cost_L, Li), t.mul(cost_D, Di))
    noise = t.mul(sigma, t.randn_like(cost_without_noise)) # N(0,sigma) noise
    cost  = t.mul(size_multiplier,cost_without_noise) + noise

    # ensure this does not ressurrects bankrupt banks
    # if size = 0, then the bank is bankrupt, so cost = 0. 
    bankrupt_mask = t.eq(size, t.tensor(0.0))
    cost = t.where(bankrupt_mask, t.tensor(0.0), cost)

    # Note: cost can be negative, but this is not a problem. 
    return cost

# try method 1 from Deep Learning for solving dynamic economic models.
# by Lilia Maliar, Serguei Maliar, Pablo Winant 

# E is exogenous
# we need 1 neural network to approximate the loans L
# D gets determined by credit regulations D/(M+L) <= lmda
# the last equation is equivalent to D <= E lmda/(1-lmda)
# the reserves M will be determined by accounting M+L = D+E 

#TODO: this neural network outputs many 0s because the linear combination of
# many relus is relatively likely to be 0.
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
        x = t.sigmoid(self.fc3(x))
        return x
hidden_size = 16
percent_assets_to_reserves = PolicyNet(input_size=N_banks,
                      hidden_size=hidden_size,
                      output_size=N_banks)

# history for debugging
hist_E = []
hist_L = []
hist_D = []
hist_M = []
append_hist = lambda history, event: history.append(event.detach().numpy())

def next_equity_size_and_dividents(Ei, size):
    """
    This function computes the next equity, size and dividends of the bank.
    It takes the Equity as imput, inferrs the Deposits from the capital 
    requirement, and once the total assets (D+E) are known, it computes the
    proportion of assets that are reserves, and then the loans. 
    """
    Di = Ei*lmda/(1-lmda)
    total_assets = Di + Ei
    Mi = t.mul(percent_assets_to_reserves(Ei), total_assets) ## NN in action

    # Li = Di + Ei - Mi
    Li = t.add(t.add(Di, Ei), -Mi)

    # profits = Mi*rM + Li*rL(Li) - Di*rD(Di) - cost(Li,Di)
    profits = t.add(t.add(t.mul(Mi, rM),
                    t.mul(Li, rL(Li))),
                    t.add(-t.mul(Di, rD(Di)), 
                    -cost(Li,Di, size)))

    dividends = t.mul(propor_div, profits)
    # equity_next = Ei + profits - dividends
    equity_next = t.add(t.add(Ei, profits), -dividends)    
    # if equity negative, the bank is bankrupt forever
    equity_next = t.max(equity_next, t.tensor(0.0))
    
    # ROE = t.div(profits, Ei) # around 10%, close to empirical ROE

    size_next = size_of_bank_after_mergers(size, equity_next)

    return equity_next, size_next, dividends
 
def objective():
    value = t.zeros(N_banks)

    n_simulations_in_epoch=10
    for _ in range(n_simulations_in_epoch):
        Ei = t.mul(t.rand(N_banks), MAX_EQUITY) # initial equity
        size = t.distributions.dirichlet.Dirichlet(t.ones(N_banks)).sample() * N_banks
        sdf_t = 1.0
        for _ in range(T):
            Ei, size, dividends = next_equity_size_and_dividents(Ei, size) # update Ei
            # value += sdf_t*dividends 
            value = t.add(value, t.mul(sdf_t, dividends))
            sdf_t *= SDF # TODO: add stochastic discount factor
    # divide the value by the number of simulations to get the average
    value = t.div(value, n_simulations_in_epoch)
    return t.mean(value) # the social planner cares equally about all banks

optimizer = optim.Adam(percent_assets_to_reserves.parameters(), lr=0.001)

def initialize_and_load_NN():
    try: 
        percent_assets_to_reserves = PolicyNet(input_size=N_banks,
                                    hidden_size=hidden_size,
                                    output_size=N_banks)
    
        percent_assets_to_reserves.load_state_dict(t.load('percent_assets_to_reserves.pt'))
        return percent_assets_to_reserves
    except: 
        print("no weights found for the NN. Did you run train_NN.py?")


losses = []