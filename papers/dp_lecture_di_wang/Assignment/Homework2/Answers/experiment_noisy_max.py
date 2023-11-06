#!/usr/bin/env python3
# Konstantin Burlachenko. Assigment 2.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import threading
import math

# Plotting configuration
plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.size"] = 12
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
legend_font_size = 20

# Available set of prices from which
clientPrices = [0.01, 0.10, 0.20, 0.30, 0.50] # Domain for client prices
sellPrices   = [0.08, 0.12, 0.25, 0.35, 0.45, 0.50] # Domain for selling prices


def datasetX(n):
    '''Generate dataset with random prices for the item by 'n' individuals with indicies into prices''' 
    #np.random.seed(123)
    if n == 8:
        pricesDataset = np.array([0,1,2,3,4,4,2,0])
    else:
        print("APPEND CODE TO HANDLE {n} ITEMS IN DATASET")
#    pricesDataset = np.random.randint(0, len(clientPrices), n)
    return pricesDataset

def scoreY(dx, priceSellIndex):
    '''Score function would be the revenue obtained from that set of individuals at a given price
        dx - dataset x that contains (private) information about assets cost (inidicies into prices)
        priceIndex - price at which we sell asset (priceIndex)
    '''

    s = 0.0
    dx = dx[:]
    p = sellPrices[priceSellIndex]

    for i in range(len(dx)):
        if clientPrices[dx[i]] >= p:
            # if client is ready product by price at price even higher then "p" then:
            # - client will buy it for lower price as well
            # - we as a seller will obtain revenue "p" from that
            s += p
    return s

def advance_price_selector(xSelectedPrices, total_prices):
    '''Advance price selector to one position'''
    cf = 1
    for i in range(len(xSelectedPrices) - 1, -1, -1):
        xSelectedPrices[i] = xSelectedPrices[i] + cf
        if xSelectedPrices[i] >= total_prices:
            xSelectedPrices[i] = 0
            cf = 1
        else:
            cf = 0

gs_cache = {}

class WorkerThread(threading.Thread):
    def __init__(self, client_price, n):
        threading.Thread.__init__(self)

        self.client_price = client_price
        self.n = n
        self.gs = 0

    def run(self):
        n = self.n
        gs = 0.0
        prices_count = len(clientPrices)
        total_possible_datasets = prices_count**n
        #==============================================================================================================
        for y_price_index in [self.client_price]:
            d_selector = [0] * n

            for t in range(total_possible_datasets):
                q = scoreY(d_selector, y_price_index)

                for pi in range(prices_count):
                    for dx_i in range(len(d_selector)):
                        p_prev = d_selector[dx_i]

                        d_selector[dx_i] = pi
                        q_prime = scoreY(d_selector, y_price_index)
                        d_selector[dx_i] = p_prev

                        delta = abs(q_prime - q)

                        if delta > gs:
                            gs = delta

                # Move to next dataset price assigment
                advance_price_selector(d_selector, prices_count)
        #==============================================================================================================
        self.gs = gs

def computeGS(n):
    '''Compute GS for a dataset of a fixed size by definition
        \sup_y \sup_{D~D'} |q(y,D) - q(y,D')|
    '''
    if n in gs_cache:
        return gs_cache[n]
    threads = []

    for sell_price_index in range(len(sellPrices)):
        threads.append(WorkerThread(sell_price_index, n))

    for th in threads:
        th.start()

    for th in threads:
        th.join()

    gs = 0
    for th in threads:
        gs = max(gs, th.gs)

    gs_cache[n] = gs
    return gs


def noisy_max(d, gs, eps):
    lamb = 2*gs/eps                       # Find lamba (or b or scale) for Laplacian distribution
    max_score_value, max_score_index = max([ (scoreY(d, si) + np.random.exponential(scale = lamb), si) for si in range(len(sellPrices))])
    return scoreY(d, max_score_index)

def exp_mechanism(d, gs, eps):
    sample_probabilities = [math.exp(scoreY(d, si)*(eps)/(2.0*gs)) for si in range(len(sellPrices))]
    s = sum(sample_probabilities)
    sample_probabilities = [sample_probabilities[i] / s for i in range(len(sample_probabilities))]
    max_score = np.random.choice(np.arange(len(sellPrices)), p=sample_probabilities)
    # Convert score index into actual gained score
    return scoreY(d, max_score)

#======================================================================================
def experiment_with_noisy_max(sizes_n, score_estimator, epsilons, experiments = 900, xInTest = None):
    f_plus_sigma  = np.zeros(len(sizes_n) * len(epsilons))
    f_minus_sigma = np.zeros(len(sizes_n) * len(epsilons))
    f_exp    = np.zeros(len(sizes_n) * len(epsilons))
    f_stddev = np.zeros(len(sizes_n) * len(epsilons))
    f_real = np.zeros(len(sizes_n) * len(epsilons))

    i = 0
    for n in sizes_n:
        gs = computeGS(n)
        d = datasetX(n)
        max_score, max_score_index = max([(scoreY(d, si), si) for si in range(len(sellPrices))])

        for eps in epsilons:
            fMean = 0.0
            fSqrMean = 0.0

            # Launch several experiments with fixed: x, eps, n
            for e in range(experiments):
                r_f = (max_score - score_estimator(d, gs, eps))
                fMean += r_f
                fSqrMean += r_f*r_f

            fMean /= experiments
            fSqrMean /= experiments
            fStdDev = (fSqrMean - (fMean)**2)**0.5

            f_plus_sigma[i] = fMean + 1.0 * fStdDev
            f_minus_sigma[i] = fMean - 1.0 * fStdDev
            f_exp[i] = fMean
            f_stddev[i] = fStdDev
            f_real[i] = max_score
            i += 1

    return f_plus_sigma, f_minus_sigma, f_exp, f_stddev
    #          0               1           2       3
#======================================================================================
print("1. Analyze dependency on epsilon")
#======================================================================================
color = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999", "#ffff33"]
linestyle = ["solid", "solid", "solid", "solid","solid","solid", "solid", "solid", "solid"]

if True:
    n_input = [8]
    epsilons = np.arange(0.001, 10.0, 0.01)

    np.random.seed(1)
    a_plus_sigma, a_minus_sigma, a_exp, a_stddev     = experiment_with_noisy_max(n_input, noisy_max, epsilons = epsilons)
    np.random.seed(1)
    b_plus_sigma, b_minus_sigma, b_exp, b_stddev = experiment_with_noisy_max(n_input, exp_mechanism, epsilons = epsilons)
    #=========================================================================================================================
    variance_relative_n     = plt.figure(figsize=(9, 12))
    fig = plt.figure(variance_relative_n.number)

    ax = fig.add_subplot(3, 1, 1)
    graphic = 4
    ax.plot(epsilons, a_exp, color=color[graphic], linestyle=linestyle[graphic], label = "Noisy-Max Mechanism, $E[q_{max} - q(y,D)] \\pm \\hat{\\sigma}$")
    ax.fill_between(epsilons, a_minus_sigma, a_plus_sigma, alpha=0.2, color=color[graphic])
    ax.grid(True)
    ax.legend(loc='best', fontsize = legend_font_size)
    ax.set_ylabel('$q_{max} - q(y,D)$')

    ax = fig.add_subplot(3, 1, 2)
    graphic = 5
    ax.plot(epsilons, b_exp, color=color[graphic], linestyle=linestyle[graphic], label = "Exponential Mechanism, $E[q_{max} - q(y,D)] \\pm \\hat{\\sigma}$")
    ax.fill_between(epsilons, b_minus_sigma, b_plus_sigma, alpha=0.2, color=color[graphic])
    ax.grid(True)
    ax.legend(loc='best', fontsize = legend_font_size)
    ax.set_ylabel('$q_{max} - q(y,D)$')

    ax = fig.add_subplot(3, 1, 3)
    graphic = 6
    ax.plot(epsilons, a_stddev, color=color[graphic],  linestyle=linestyle[graphic], label = "Noisy-Max Mechanism")
    graphic = 7
    ax.plot(epsilons, b_stddev, color=color[graphic], linestyle=linestyle[graphic], label = "Exponential Mechanism")
    ax.grid(True)
    ax.legend(loc='best', fontsize = legend_font_size)
    ax.set_xlabel('$\\varepsilon$')
    ax.set_ylabel('$\\sqrt{\\hat{Var}[q_{max} - q(y,D)]}$')
    fig.tight_layout()
    plt.show()
    print()
    print("GS cache")
    print(gs_cache)
#======================================================================================

