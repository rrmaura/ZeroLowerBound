import torch 
from simple_definitions import *

initial_deposits = 1
TODAY = 1
# get a sense of the rD. Plot it with different values of initial deposits
def f(time, total_D):
    total_D = torch.tensor(total_D)
    Y = Y0 * (1+g_y) ** time 


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

# plot f as a function of total D for time = 1
import matplotlib.pyplot as plt
import numpy as np

total_D = np.linspace(1, GDP(TODAY)-500, 100)
rD = f(TODAY, total_D)
# plt.plot(total_D, rD)
# plt.show()


# plot profits for the monopoly as as function of total D for time = 1
def profits_monopoly(D):
    E = 1
    rD = f(TODAY, D)
    M = E + D 
    profits = M*rM - D*rD.numpy()
    return profits

# also plot a vertical line in the solution with optimal monopoly depostits
Rm = 1 + rM 
T =  1/(Rm*beta**2) # we would need to get T = 0.26 to have reasonable rD. 
# this is unfeasible, we need to change theta! 
propor_savings_solution = (1+T)-np.sqrt(T*(2+T))
# monopoly_deposits = (Rm / (2*(Rm + (1/beta**2)))) * GDP(TODAY)
monopoly_deposits = propor_savings_solution * GDP(TODAY)
print("monopoly deposits: ", monopoly_deposits)
rD_monopoly = f(TODAY, monopoly_deposits)
print("rD monopoly: ", rD_monopoly)

Deposits = SimplePolicyNet(input_size=N_banks+1,
                                hidden_size=hidden_size,
                                output_size=N_banks)
    

Deposits.load_state_dict(t.load('Deposits.pt'))
time_tensor = t.tensor([TODAY])
input = t.cat((t.ones(1), time_tensor), dim=-1)
NN_deposits = Deposits(input) * GDP(TODAY)

rD_NN = f(TODAY, NN_deposits)


profits = profits_monopoly(total_D)
interst_rate = f(TODAY, total_D)

print(rD_monopoly)
plt.plot(total_D, profits)
plt.plot(total_D, interst_rate)
plt.axvline(x=monopoly_deposits, color='r', linestyle='--')
plt.axvline(x=NN_deposits.detach().numpy()[0], color='y', linestyle='--')
# axis 
plt.xlabel('Total deposits')
plt.ylabel('Profits')
# legend
plt.legend(['Profits', 'Interest rate', 'Monopoly deposits', 'NN deposits'])
plt.show()

plt.plot(total_D, interst_rate)
plt.show()



# plt.plot(total_D, profits)
# plt.show()