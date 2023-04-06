# TODO: Why do people perfer Loans to Deposits? Deposits higher return and no cost
# TODO: bigger neural networks? 
# TODO: the deposits are touching the lower bound
# TODO: all companies should be symmetric in the beginning. use last layer to enforce this? 
# TODO: explosive growht. Companies increase size by 100% in 5 years
# TODO: equity deposits rations unstables
# TODO: Profiling for faster code. 
# TODO: what is Y in the household problem? 
# TODO peace of mind. Ensure that without any shocks,
#  the system is stable and the solution is the analytical one
# TODO: add shocks to the system
# TODO: so far, ratio of deposits to assets is equality. Make it inequality.
# TODO: add things to assert. E.g. ROE around 5% and 10%. Rates not 1000%.
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
t.manual_seed(1)
np.random.seed(1)
random.seed(1)

percent_assets_to_reserves = initialize_and_load_NN()

# plot the policy function for the first bank from E = 0 to E = INITIAL_EQUITY
# assume all other banks have the same equity = INITIAL_EQUITY
n_points = 1000
E = t.ones(n_points, N_banks) * INITIAL_EQUITY
E[:,0] = t.linspace(0,INITIAL_EQUITY,n_points)
total_assets = t.add(E, E*lmda/(1-lmda))
M = t.mul(percent_assets_to_reserves(E), total_assets)
equity_first_bank = E[:,0].detach().numpy()
reserves_first_bank = M[:,0].detach().numpy()

# plot
plt.plot(equity_first_bank, reserves_first_bank)
plt.title("Reserves as a function of equity for the first bank")
plt.savefig('plots/reserves_first_bank.png')
plt.clf()

# run a simulation
length_simulation = 25

# history 
hist_totalAssets = np.zeros((length_simulation, N_banks))
hist_E = np.zeros((length_simulation, N_banks))
hist_L = np.zeros((length_simulation, N_banks))
hist_D = np.zeros((length_simulation, N_banks))
hist_M = np.zeros((length_simulation, N_banks))
hist_rL = np.zeros(length_simulation)
hist_rD = np.zeros(length_simulation)
hist_rM = np.ones(length_simulation) * rM

# initial equity and size
E = t.ones(N_banks)
size = t.ones(N_banks)
sum_size = size.sum()
for i in tqdm(range(length_simulation)):
    # ensure simulation space is same as training space
    assert (E >= 0).all()
    assert (MAX_EQUITY >= E).all() 
    assert sum_size == size.sum() # sum size should stay constant 
    E, size, dividends = next_equity_size_and_dividents(E, size) # update E
    D = E*lmda/(1-lmda)
    total_assets = t.add(E, D)
    M = t.mul(percent_assets_to_reserves(E), total_assets)
    L = t.add(t.add(E, E*lmda/(1-lmda)), -M)

    # append history
    hist_totalAssets[i] = total_assets.detach().numpy()
    hist_E[i] = E.detach().numpy()
    hist_L[i] = L.detach().numpy()
    hist_D[i] = D.detach().numpy()
    hist_M[i] = M.detach().numpy()
    hist_rL[i] = rL(L).detach().numpy()
    hist_rD[i] = rD(D).detach().numpy()

accounting_condition = (hist_totalAssets - hist_M - hist_L)
assert np.isclose(accounting_condition,0, 0.0001, 0.0001).all

#######################################################
                        # PLOTS #
#######################################################

# plot the distribution of equity at time 0, T/2 and T
for time in [0, length_simulation//2, length_simulation-1]:
    plt.hist(hist_E[time], bins=100)
    plt.title('distribution of equity at time ' + str(time))
    plt.savefig('plots/distribution_of_equity_at_time_' + str(time) + '.png')
    plt.clf()
    
# plot the mean history of Equity and Loans
for account, hist_acc in [("Equity", hist_E),
                         ("Loans", hist_L), 
                         ("Deposits", hist_D), 
                         ("Reserves", hist_M), 
                         ("total assets", hist_totalAssets)]:
    
    # plot the history of E for the first bank
    plt.plot(hist_acc[:,0])
    title = "first bank's " + account + " over time"
    plt.title(title)
    plt.savefig(f"plots/{title.lower().replace(' ', '_')}.png")
    plt.clf()

    # plot the mean history of Account
    hist_acc_mean = hist_acc.mean(axis=1)
    plt.plot(hist_acc_mean)
    title = "mean " + account + " over time"
    plt.title(title)
    plt.savefig(f"plots/{title.lower().replace(' ', '_')}.png")
    plt.clf()

# Empirically, a bank's accounting looks like this:
###########################################
#                    #                    #
#    LOANS           #     EQUITY         #                
#    95%             #     10-20%         #                 
#                    #     (15%)          #          
#                    #                    #          
#                    ######################                              
#                    #                    #          
#                    #      DEPOSITS      #          
#                    #       80-90%       #          
#                    #       (85%)        #          
#                    #                    #          
#                    #                    #          
######################                    #         
#      MONETARY      #                    #          
#      RESERVES      #                    #          
#        <5%         #                    #          
###########################################

# rational ratios
ideal_ratio_E_D = [0.15/0.85] * length_simulation
ideal_ratio_M_L = [0.05/0.95] * length_simulation

# plot, for the first firm, the ratios 
ratio_E_D = hist_E[:,0]/hist_D[:,0]
ratio_M_L = hist_M[:,0]/hist_L[:,0]

# D is deterministic given E, lmda = 0.85, so this ratio is always correct
time = np.arange(length_simulation)
plt.plot(time, ratio_E_D, 'b--' , label = 'E/D')
plt.plot(time, ideal_ratio_E_D, 'b-' , label = 'ideal E/D')
plt.legend(loc="upper left")
plt.title('ratios')
plt.ylim(0,0.5) # plot has Y axis between 0 and 0.5
plt.savefig("plots/first_firm's_accounting_E_D_ratios.png")
plt.clf()

plt.plot(time, ratio_M_L, 'r--' , label = 'M/L')
plt.plot(time, ideal_ratio_M_L, 'r-' , label = 'ideal M/L')
plt.legend(loc="upper left")
plt.title('ratios')
plt.savefig("plots/first_firm's_accounting_M_L_ratios.png")
plt.clf()

# check that the interest rates evolve together. Plot all 3 interest rates
# are rates of deposits higher or lower than rates of central bank? 
hist_rM_array = np.array(hist_rM)
hist_rL_array = np.array(hist_rL)
hist_rD_array = np.array(hist_rD)

plt.plot(time, hist_rM_array,       'b--'   , label = 'rM')
plt.plot(time, hist_rL_array,       'r--'   , label = 'rL')
plt.plot(time, hist_rD_array,       'g--'   , label = 'rD')
plt.legend(loc="upper left")
plt.title('interest rates')
plt.savefig('plots/interest_rates.png')
plt.clf()

# we can do alternative method (XGBoost) and see that solutions coincide. 
# TODO: check maliars and how they did it

# see how the model evolves (retraining the NN) with r_m increasing from -0.1 to 0.1

print("\n \n done \n \n")

