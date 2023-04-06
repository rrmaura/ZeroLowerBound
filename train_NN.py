"""
this program is a simulation of a macroeconomic model. This
is a partial equilibrium model, with only heteorgeneous banks. 
Banks can choose Loans (L), Deposits (D), Reserves (M) and Equity (E)
such that M+L = D+E
"""
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from definitions import *

# seeds
t.manual_seed(0)
np.random.seed(0)
random.seed(0)

n_epochs = 600 # number of epochs
patience = 200  # number of epochs to wait before early stopping

best_loss = float('inf')
wait = 0  # number of epochs waited
for epoch in tqdm(range(n_epochs)):
    # keep track of the training error
    optimizer.zero_grad()
    loss = -objective() # we want to maximize the objective
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch} loss: {loss}")

    # Early stop. There is no need for validation set.
    # check if loss improved
    if loss < best_loss:
        best_loss = loss
        wait = 0
        # save the neural network weights
        t.save(percent_assets_to_reserves.state_dict(), 'percent_assets_to_reserves.pt')

    else:
        wait += 1
        if wait >= patience:
            print(f"Loss hasn't improved in {patience} epochs. Stop")
            # load the best weights
            percent_assets_to_reserves.load_state_dict(t.load('percent_assets_to_reserves.pt'))
            break

# save the neural network weights
t.save(percent_assets_to_reserves.state_dict(), 'percent_assets_to_reserves.pt')


# plot the training error and save in plots folder
plt.plot(losses)
plt.title("losses")
plt.show()
plt.plot(losses)
plt.title("losses")
plt.savefig('plots/losses.png')
