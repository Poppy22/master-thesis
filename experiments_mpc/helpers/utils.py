import numpy as np
import torch
import matplotlib.pyplot as plt

# ================== Function to sample the input ==================
def sample_dataset(n, p, c, k, r):
    np.random.seed(0)
    x = np.random.uniform(-r, r, (n, p))
    noise = np.random.uniform(-c, c, n)
    
    # theta_star = np.random.uniform(-r, r, p)
    theta_star = np.random.normal(0, r, p)

    # make theta_star k-sparse randomly
    if k > 0 and k < p:
        index = np.random.choice(range(p), p - k, replace=False)
        for i in index:
            theta_star[i] = 0

    # normalize theta_star
    norm = np.linalg.norm(theta_star, 2)
    theta_star = theta_star / norm
        
    y = x @ theta_star + noise
    return (torch.tensor(x), torch.tensor(y), torch.tensor(theta_star))


# ==================== DICT SINGLE KEY ======================
# dictionary key: integer
# dictionary value: an array (the y values)
# the x values: iteration number [0 ... T]
def plot_error_by_iteration_and_dict_key(d, title, xlabel, ylabel, line_label):
    plt.figure(figsize=(10, 10))

    style = {
        'private': 'dashed',
        'non-private': 'solid'
    }

    colors = []
    for c in range(int(len(d.keys()) / 2)):
        colors.append((np.random.random(), np.random.random(), np.random.random()))

    i = 0
    increase_color = False
    for n in d.keys():
        T = len(d[n])
        plt.plot(range(T), d[n], linewidth=3, label=f'{line_label}={n}')

        if increase_color:
            i += 1
        increase_color = not increase_color
    
    plt.legend(loc=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# ==================== DICT KEY TUPLE ======================
# dictionary key: tuple (n, k) or (n, p)
# dictionary value: a value (relative error)
# the n values
def plot_relative_error(results, title, xlabel, ylabel, line_label):

    for key in results.keys():
        n_values = [x[0] for x in results[key]]
        rel_error_values = [x[1] for x in results[key]]
        plt.plot(n_values, rel_error_values, label=f'{line_label}={key}', color=(np.random.random(), np.random.random(), np.random.random()))

    plt.title(title)
    plt.legend(loc=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
        
    plt.show()


# ==================== DICT KEY TUPLE ======================
# dictionary key: tuple (n, k) or (n, p)
# dictionary value: a value (relative error)
# the n values
def plot_error_by_iterations_dict_tuple(results):
    _, axs = plt.subplots(2, 2, figsize=(14, 10))

    r, c = 0, 0

    values_1 = [key[0] for key in results.keys()]
    values_2 = [key[1] for key in results.keys()]

    for i in values_2:
        for key in results.keys():
            if key[1] == i:
                T = len(results[key][0])
                n, k = key
                axs[r, c].plot(range(T), results[key][0].numpy(), label=f'n={n}',
                         color=(np.random.random(), np.random.random(), np.random.random()))

        axs[r, c].set_title(f'p = {64}, k = {10}, c = {i}')
        axs[r, c].legend(loc=1)
        
        c += 1
        if c == 2:
            c = 0
            r += 1

    for ax in axs.flat:
        ax.set(xlabel='T (iterations)', ylabel='||theta_t - theta*||2')
        
    plt.show()