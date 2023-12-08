import torch
import numpy as np
import copy

torch.set_num_threads(1)
torch.set_printoptions(precision=16) # default


# Peeling (central model implementation)
# Received a vector x and a number k
# Return x made k-sparse, in a differentially private way if dp is true; else, with no noise involved
def generate_laplace_noise(lmb, ebs, dta, s, d):
    s = max(s, 1) # to avoid generating zero noise when the sparsity is 0
    scale = (lmb / ebs) * (2 * np.sqrt(3 * s * np.log(1 / dta)))
    noise = np.random.laplace(0, scale, d)
    return noise


def peeling(x, k, lmb, ebs, dlt, dp=True):
    # make a deep copy to keep the real x for the return
    x_copy = copy.deepcopy(x)

    x_copy = x_copy.abs()
    min_value = x_copy.min()
    S = set()
    d = len(x)

    for _ in range(k):
        if dp:
            noise = generate_laplace_noise(lmb, ebs, dlt, k, d)
        else:
            noise = np.zeros(d)
        x_noise = x_copy + noise # adding new noise for every iteration

        index = x_noise.argmax()
        x_copy[index] = min_value # to avoid finding the same index again
        S.add(index.item())

    # make the vector x k-sparse
    for i in range(d):
        if i not in S:
            x[i] = 0

    # add noise to the final result
    if dp:
        noise = generate_laplace_noise(lmb, ebs, dlt, k, d)
        return x + noise

    return x



def solve(data, k, T, eta, noise_params):

    x, y, theta_star = data # unpack data
    lmb, ebs, dlt, dp = noise_params # unpack noise params
    n = len(x)
    p = len(x[0])
    theta_prev = torch.zeros(p)

    iteration_error = [0]
    for t in range(T):
        all_gradients = []

        theta_prev = theta_prev.double()
        for i in range(n):
            var1 = theta_prev.dot(x[i])
            var1 = var1 - y[i]
            var2 = x[i].t()
            nabla_i = var2 * var1
            all_gradients.append(nabla_i)

        nabla_prev = torch.stack(all_gradients).sum(axis=0)
        nabla_prev = nabla_prev /  len(all_gradients)
        theta_t = theta_prev - eta * nabla_prev

        # peeling - make k-sparse
        if k > 0 and k < p:
            theta_t = peeling(theta_t, k, lmb, ebs / T, dlt / T, dp)
        elif dp:
            noise = generate_laplace_noise(lmb, ebs, dlt, k, p)
            theta_t = theta_t + noise

        e = torch.norm(theta_t - theta_star, p=2)
        diff = abs(e - iteration_error[-1])

        iteration_error.append(e.item())
        theta_prev = theta_t

        if diff <= 0.000001:
            break

    rel_error = iteration_error[-1] / torch.norm(theta_star, p=2)
    return iteration_error[1:], rel_error
