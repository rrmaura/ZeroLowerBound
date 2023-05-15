# this model is a simpler version where loans do not exists and Deposits 
# are the only choice. Make this program self contained: 

"""
this program is a simulation of a macroeconomic model. This
is a partial equilibrium model, with only heteorgeneous banks. 
Banks can choose Loans (L), Deposits (D), Reserves (M) and Equity (E)
such that M+L = D+E
"""
# why aren't all banks the same? 

# TODO: parameters - calibration?
# is INITIAL_EQUITY = 1 reasonable? How many banks? this affectgs the amount of loans

# stochastic discount factor
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
from tqdm import tqdm
import matplotlib.pyplot as plt
from simple_definitions import *



# seeds
t.manual_seed(0)
np.random.seed(0)
random.seed(0)

Deposits = SimplePolicyNet(input_size=N_banks + 1, # include time
                      hidden_size=hidden_size,
                      output_size=N_banks)

append_hist = lambda history, event: history.append(event.detach().numpy())

#####################################################################
#####################################################################
#####################################################################
#                        TRAIN                                      #
#####################################################################
#####################################################################
#####################################################################

 
# seeds
t.manual_seed(0)
np.random.seed(0)
random.seed(0)

n_epochs = 400 # number of epochs
patience = 200  # number of epochs to wait before early stopping

optimizer = optim.Adam(Deposits.parameters(), lr=0.001)


def objective():
    value = t.zeros(N_banks)

    n_simulations_in_epoch=10
    for _ in range(n_simulations_in_epoch):
        # random initialization 
        Ei = t.mul(t.rand(N_banks), MAX_EQUITY) # initial equity
        size = t.distributions.dirichlet.Dirichlet(t.ones(N_banks)).sample() * N_banks 
        # # symmetric deterministic initialization
        # Ei = t.ones(N_banks) * INITIAL_EQUITY
        # size = t.ones(N_banks) * (1.0/N_banks)
        sdf_t = 1.0
        for time in range(MAX_TIME):
            Ei_previous = Ei
            size_previous =  size # for debug

            Ei, Di, Mi, size, dividends = next_E_D_M_size_and_dividents(Ei, size, time, Deposits) 

            # value += sdf_t*dividends 
            value = t.add(value, t.mul(sdf_t, dividends))
            sdf_t *= SDF # TODO: add stochastic discount factor
    # divide the value by the number of simulations to get the average
    value = t.div(value, n_simulations_in_epoch)
    return t.mean(value) # the social planner cares equally about all banks


######################################################################
######################################################################
######################################################################
######################################################################

# train neural network 

best_loss = float('inf')
losses = []
wait = 0  # number of epochs waited
for epoch in tqdm(range(n_epochs)):
    # keep track of the training error
    optimizer.zero_grad()
    loss = -objective() # we want to maximize the objective
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch} loss: {loss.item()}")

    # Early stop. There is no need for validation set.
    # keep weights of the best model
    # check if loss improved
    if loss < best_loss:
        best_loss = loss
        wait = 0
        # weights of the best model
        t.save(Deposits.state_dict(), 'Deposits.pt')
    else:
        wait += 1
        if wait >= patience:
            print(f"Loss hasn't improved in {patience} epochs. Stop")
            # load the best weights
            Deposits.load_state_dict(t.load('Deposits.pt'))
            break

# save the neural network weights
t.save(Deposits.state_dict(), 'Deposits.pt')


# plot the training error and save in plots folder
plt.plot(losses)
plt.title("losses")
plt.show()
plt.plot(losses)
plt.title("losses")
plt.savefig('plots/losses.png')
