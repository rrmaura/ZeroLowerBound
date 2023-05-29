import torch as t
# from simple_definitions import *

beta = 0.9
rM = 0.02
def f(proport_Y, theta_2):
    proport_Y = t.tensor(proport_Y)
    Y = t.tensor(1)
    D = t.mul(proport_Y,Y)

    # The formula of returns for the firm implies:
    # r**((theta-1)/(theta)) = beta**(1/theta) * (Y-D)/D 
    # or 
    # r = (  beta**(1/theta) * (Y-D)/D  )**(theta/(theta-1))
    # or 
    # r = c * base**expon 
    # where 
    # c = beta**1/(theta-1),  expon=(theta/(theta-1)), base = (Y-D)/D

    exponent = theta_2/(theta_2-1)
    constant =  beta**(1/(theta_2-1))
    base = t.add(t.div(Y, D),-1)

    rD = t.mul(constant, t.pow(base, exponent)) - 1

    # zero lower bound
    # rD = t.max(rD, t.tensor(0.0)) 
    # rD = bound_rates(rD)
    return rD

# I want to see the rD as a function of total_D for different values of theta_2
# theta_2 = takes values between 0 and 1
# total_D takes values between 1 and GDP(TODAY)-500

import matplotlib.pyplot as plt
import numpy as np


# plot profits for the monopoly as as function of total D for time = 1 and theta_2=0.9
def profits_monopoly(proportion_Y, theta_2):
    Y = 1
    D = proportion_Y*Y
    E = 1
    rD = f( D, theta_2=theta_2)
    M = E + D 
    profits = M*rM - D*rD.numpy()
    return profits

# find proportion of GDP that maximizes profits (profits is convex function of proportion)
import scipy.optimize as opt
def minus_profits_monopoly(proportion_Y, theta_2):
    return -profits_monopoly(proportion_Y, theta_2)
for theta_2 in np.linspace(0.001, 0.999, 10):
    
    proportion_Y_max = opt.minimize_scalar(minus_profits_monopoly, args=(theta_2), bounds=(0.01,0.99), method='bounded')
    print("theta_2: ", theta_2)
    print("proportion_Y_max: ", proportion_Y_max)
    print("profits_monopoly(proportion_Y_max): ", profits_monopoly(proportion_Y_max.x, theta_2))
    print("interest rate: ", f(proportion_Y_max.x, theta_2))
    print("\n")



proportions = np.linspace(0.3, 1, 100)
theta_2 = 0.001
rD = f( proportions, theta_2=theta_2)
# name graph
plt.title("interest rates when theta_2 = " + str(theta_2))
plt.plot(proportions, rD)
plt.show()




