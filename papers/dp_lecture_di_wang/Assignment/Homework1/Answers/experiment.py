#!/usr/bin/env python3
# Konstantin Burlachenko. Assigment 1.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.size"] = 12
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
legend_font_size = 20

def f(x):
    '''Original function value of which we hide with DP'''
    return x.sum() / x.size

def randomResponse(x, eps):
    '''Random response esp-DP mechanism for average of xi={0,1}'''
    probabilities = np.exp(eps*np.ones_like(x))/(np.exp(eps*np.ones_like(x)) + 1.0) # select probabilities
    selectors = (np.random.random(x.shape) < probabilities).astype(float)           # selectors[i]=1.0 w.p. probabilities[i] 
    r = x * selectors + (1-x) * (1-selectors)                                       # ri = x[i] w.p. probabilities[i] and 1-x[i] w.p. 1.0 - probabilities[i]

    z = (np.exp(eps) + 1)/(np.exp(eps) - 1) * r - 1.0/(np.exp(eps) - 1.0)
    return z.sum() / z.size

def laplaceMechanism(x, eps):
    '''Laplace esp-DP mechanism for average of xi={0,1}'''
    GS = 1.0/x.size   # Global sensitivity for average query
    lamb = GS/eps     # Find lamba (or b or scale) for Laplacian distribution
    Y = np.random.laplace(scale = lamb) # Generate r.v. with Lap(lamb)
    return x.sum() / x.size + Y         # Response

#======================================================================================
def experiment_with_dp(sizes_n, func, epsilons, experiments = 900, xInTest = None):
    f_plus_sigma  = np.zeros(len(sizes_n) * len(epsilons))
    f_minus_sigma = np.zeros(len(sizes_n) * len(epsilons))
    f_exp    = np.zeros(len(sizes_n) * len(epsilons))
    f_stddev = np.zeros(len(sizes_n) * len(epsilons))
    f_real = np.zeros(len(sizes_n) * len(epsilons))

    i = 0
    for n in sizes_n:
        for eps in epsilons:
            fMean = 0.0
            fSqrMean = 0.0
            if xInTest is not None:
                xin = xInTest
            else:
                xin = (np.random.rand(n) > 0.5).astype(float)  # Generate input as {0,1}**n u.a.r

            # Launch several experiments with fixed: x, eps, n
            for e in range(experiments):
                r_f = (func(xin, eps) - f(xin))
                fMean += r_f
                fSqrMean += r_f*r_f

            fMean /= experiments
            fSqrMean /= experiments
            fStdDev = (fSqrMean - (fMean)**2)**0.5

            f_plus_sigma[i] = fMean + 1.0 * fStdDev
            f_minus_sigma[i] = fMean - 1.0 * fStdDev
            f_exp[i] = fMean
            f_stddev[i] = fStdDev
            f_real[i] = f(xin)
            i += 1

    return f_plus_sigma, f_minus_sigma, f_exp, f_stddev
    #          0               1           2       3
#======================================================================================
print("1. Check unbiasedness for the fixed input")
#======================================================================================

if True:
    np.random.seed(1)
    xInTest = np.ones(100)
    print("Random response expectation of error based on 1000 experiments: ", experiment_with_dp([100], randomResponse, epsilons=[0.1], experiments = 1000, xInTest=xInTest)[2])
    print("Random response expectation of error based on 100000 experiments: ", experiment_with_dp([100], randomResponse, epsilons=[0.1], experiments = 100000, xInTest=xInTest)[2])
    print("Random response expectation of error based on 500000 experiments: ", experiment_with_dp([100], randomResponse, epsilons=[0.1], experiments = 500000, xInTest=xInTest)[2])
    print("")
    print("Laplacian mechanism expectation of error based on 1000 experiments: ", experiment_with_dp([100], laplaceMechanism, epsilons=[0.1], experiments = 1000, xInTest=xInTest)[2])
    print("Laplacian mechanism expectation of error based on 100000 experiments: ", experiment_with_dp([100], laplaceMechanism, epsilons=[0.1], experiments = 100000, xInTest=xInTest)[2])
    print("Laplacian mechanism expectation of error based on 500000 experiments: ", experiment_with_dp([100], laplaceMechanism, epsilons=[0.1], experiments = 500000, xInTest=xInTest)[2])
#======================================================================================
print("2. Analyze dependency on epsilon")
#======================================================================================
color = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999", "#ffff33"]
linestyle = ["solid", "solid", "solid", "solid","solid","solid", "solid", "solid", "solid"]

if True:
    n_input = [1000]
    epsilons = np.arange(0.01, 2.0, 0.01)

    np.random.seed(1)
    rr_plus_sigma, rr_minus_sigma, rr_exp, rr_stddev     = experiment_with_dp(n_input, randomResponse, epsilons = epsilons)
    np.random.seed(1)
    lap_plus_sigma, lap_minus_sigma, lap_exp, lap_stddev = experiment_with_dp(n_input, laplaceMechanism, epsilons = epsilons)
    #=========================================================================================================================
    variance_relative_n     = plt.figure(figsize=(9, 12))
    fig = plt.figure(variance_relative_n.number)

    ax = fig.add_subplot(3, 1, 1)
    graphic = 4
    ax.plot(epsilons, rr_exp, color=color[graphic], linestyle=linestyle[graphic], label = "Random Response, $E[f(x) - f_{real}(x)] \\pm \\hat{\\sigma}$")
    ax.fill_between(epsilons, rr_minus_sigma, rr_plus_sigma, alpha=0.2, color=color[graphic])
    ax.grid(True)
    ax.legend(loc='best', fontsize = legend_font_size)
    ax.set_ylabel('$f(x) - f_{real}(x)$')

    ax = fig.add_subplot(3, 1, 2)
    graphic = 5
    ax.plot(epsilons, lap_exp, color=color[graphic], linestyle=linestyle[graphic], label = "Laplacian Mechanism, $E[f(x) - f_{real}(x)] \\pm \\hat{\\sigma}$")
    ax.fill_between(epsilons, lap_minus_sigma, lap_plus_sigma, alpha=0.2, color=color[graphic])
    ax.grid(True)
    ax.legend(loc='best', fontsize = legend_font_size)
    ax.set_ylabel('$f(x) - f_{real}(x)$')

    ax = fig.add_subplot(3, 1, 3)
    graphic = 6
    ax.loglog(epsilons, rr_stddev, color=color[graphic],  linestyle=linestyle[graphic], label = "Random Response")
    graphic = 7
    ax.loglog(epsilons, lap_stddev, color=color[graphic], linestyle=linestyle[graphic], label = "Laplacian Mechanism")
    ax.grid(True)
    ax.legend(loc='best', fontsize = legend_font_size)
    ax.set_xlabel('$\\varepsilon$')
    ax.set_ylabel('$\\sqrt{\\hat{D}[f(x) - f_{real}(x)]}$')
    fig.tight_layout()
    plt.show()
#======================================================================================
print("3. Analyze dependecny on n")
#======================================================================================

if True:
    n_input = range(100, 10*1000, 100)
    np.random.seed(1)
    rr_plus_sigma, rr_minus_sigma, rr_exp, rr_stddev     = experiment_with_dp(n_input, randomResponse, epsilons = [0.001])
    np.random.seed(1)
    lap_plus_sigma, lap_minus_sigma, lap_exp, lap_stddev = experiment_with_dp(n_input, laplaceMechanism, epsilons = [0.001])
    #=========================================================================================================================
    variance_relative_n     = plt.figure(figsize=(9, 12))
    fig = plt.figure(variance_relative_n.number)

    ax = fig.add_subplot(3, 1, 1)
    graphic = 0
    ax.plot(n_input, rr_exp, color=color[graphic], linestyle=linestyle[graphic], label = "Random Response, $E[f(x) - f_{real}(x)] \\pm \\hat{\\sigma}$")
    ax.fill_between(n_input, rr_minus_sigma, rr_plus_sigma, alpha=0.2, color=color[graphic])
    ax.grid(True)
    ax.legend(loc='best', fontsize = legend_font_size)
    ax.set_ylabel('$f(x) - f_{real}(x)$')

    ax = fig.add_subplot(3, 1, 2)
    graphic = 1
    ax.plot(n_input, lap_exp, color=color[graphic], linestyle=linestyle[graphic], label = "Laplacian Mechanism, $E[f(x) - f_{real}(x)] \\pm \\hat{\\sigma}$")
    ax.fill_between(n_input, lap_minus_sigma, lap_plus_sigma, alpha=0.2, color=color[graphic])
    ax.grid(True)
    ax.legend(loc='best', fontsize = legend_font_size)
    ax.set_ylabel('$f(x) - f_{real}(x)$')

    ax = fig.add_subplot(3, 1, 3)
    graphic = 2
    ax.loglog(n_input, rr_stddev, color=color[graphic],  linestyle=linestyle[graphic], label = "Random Response")
    graphic = 3
    ax.loglog(n_input, lap_stddev, color=color[graphic], linestyle=linestyle[graphic], label = "Laplacian Mechanism")
    ax.grid(True)
    ax.legend(loc='best', fontsize = legend_font_size)
    ax.set_xlabel('n (size of input)')
    ax.set_ylabel('$\\sqrt{\\hat{D}[f(x) - f_{real}(x)]}$')
    fig.tight_layout()
    plt.show()
