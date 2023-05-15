import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Parameters:
Y0 = 1000 # 40000
g_y = 0.04 # growth rate of Y GDP # TODO: change to steady state growth D

A0 = 500 #2500
g_a = 0.01 # growth rate of A

hidden_size = 16

MAX_EQUITY = 100 # max equity used for training the NN
INITIAL_EQUITY = 1 # 35 # initial equity of the banks
MAX_TIME = 10 # max time used for training the NN
N_banks = 1 # number of banks (TODO: change it to 4236)
theta = 0.5 # risk_aversion , inverse of the elasticity of substitution
beta = 0.98 # discount factor
# TODO: add stochastic discount factor (utility function of households)
# SDF = 0.97 * beta #Stochastic discount factor 
SDF = 0.5 # we need to make value function converge
lmda = 0.85 # legal ratio of deposits to assets (0.97 in calibration, but this
# value gives a more reasonable ratio of deposits to equity )
propor_div = 0.04 # proportion paid out as dividends (0.02<mu<0.05)
rM = 0.03 # interest rate central bank
sigma = 0.00 # variance of the shock of cost (we have to think about units)
R_M = 1 + rM 
alpha = 0.3 # capital share Y = A * K**(alpha)

# we assume the cost is linear in L and D
cost_L = 0.0
cost_D = 0.0

# define functions of demand (depends on D)

def bound_rates(unbound_r): 
    """
    This function takes a rate and returns a bounded rate. 
    It is a differentiable function, to make training easier. 
    """
    low_bound = 0.0
    high_bound = 0.20
    return t.clamp(unbound_r, low_bound, high_bound)

def rD(Di, time):
    # The demand for deposits is not linear in the total amount of deposits
    # it is derived from the household maximization problem

    # exogenous formula for Y
    Y = Y0 * (1+g_y) ** time 

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

    rD = t.mul(constant, t.pow(base, exponent)) - 1

    # zero lower bound
    # rD = t.max(rD, t.tensor(0.0)) 
    # rD = bound_rates(rD)
    return rD


def R_D(Di, time):
    r_D = rD(Di,  time)
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
    total_equity = t.sum(Ei)
    
    if total_equity < 0.00001:
        # raise ValueError("total equity is 0, so we cannot divide by it")
        # everyone is bankrupt, so no one buys anyone. Size does not matter
        return size
    
    
    prob_buy_bankrupt = t.div(Ei, total_equity)
    
    new_size = t.add(size, t.mul(size_bankrupt, prob_buy_bankrupt))

    # eliminate bankrupt banks. Set their size to 0. 
    new_size = t.where(bankrupt_mask, t.tensor(0.0), new_size)
    return new_size 

def cost(Di, size):
    size_multiplier = size_multiplier_f(size) 
    cost_without_noise = t.mul(cost_D, Di)
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
class SimplePolicyNet(nn.Module):
    # 2 hidden layer neural network
    def __init__(self, input_size, hidden_size, output_size):
        super(SimplePolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = t.sigmoid(self.fc3(x))
        return x
    

def next_E_D_M_size_and_dividents(Ei, size, time, Deposits):
    """
    This function computes the next equity, size and dividends of the bank.
    It takes the Equity as imput, computes the Deposits, and once the 
    total assets (D+E) are known, it buys reserves. 
    """

    Di, Mi = Deposits_and_Reserves(Ei, time, Deposits)

    # profits = Mi*rM - Di*rD(Di)Li,  - cost(Li,Di)
    profits = t.add(t.add(t.mul(Mi, rM), -t.mul(Di, rD(Di,  time))), -cost(Di, size)) 

    dividends = t.mul(propor_div, profits)
    dividends = t.max(dividends, t.tensor(0.0)) # dividends can't be negative 

    # equity_next = Ei + profits - dividends
    equity_next = t.add(t.add(Ei, profits), -dividends)    
    equity_next = t.max(equity_next, t.tensor(0.0))# equity negative => bank is bankrupt forever
    
    # ROE = t.div(profits, Ei) # around 10%, close to empirical ROE

    size_next = size_of_bank_after_mergers(size, equity_next)

    return equity_next, Di, Mi, size_next, dividends

def Deposits_and_Reserves(Ei, time, Deposits): 
    time_tensor = t.tensor(time).view(1)
    input = t.cat((Ei, time_tensor), dim=-1)
    Di = Deposits(input) * GDP(time)
    Mi = t.add(Di, Ei)
    return Di, Mi

def GDP(time): 
    # return t.tensor(Y0 * (1 + g_y)**time)
    return t.tensor((Y0 * (1 + g_y)**time)/N_banks) # TODO: theoretically justify this

    