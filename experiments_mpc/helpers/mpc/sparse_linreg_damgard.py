import crypten
import torch
import numpy as np
import time

import crypten.mpc as mpc

torch.set_num_threads(1)
torch.set_printoptions(precision=16) # default


# ================== Function to cut c bits from each share, for efficiency ==================
def cut_c_bits(v, c):
    power_c = 2 ** c
    for i in range(len(v)):
        v[i] = v[i].div(power_c)

    return v


# ================== Function to debugging ==================
def reveal_and_print(x, info_text=''):
    x_rev = x.get_plain_text()
    crypten.print(f"Source {0} {info_text}= {x_rev}")


# ================== Function for making k-sparse ==================
# for truncation step
# returns an array of the indices that should be made zero (i.e. the smallest n - k values)
# makes use of the bits_to_cut for efficiency

def get_indices(x, noise, k, d, bits_to_cut=0):
    index_k_argmax = torch.zeros(d)
    x_copy = x.clone() # to avoid cutting bits on the real input
    
    # preparing for argmax:
    # cutting c bits for efficiency reasons
    if bits_to_cut > 0:
        x_copy = x_copy.div(bits_to_cut) # removing the last c bits from every share

    x_copy_abs = x_copy.abs()
    min_value = x_copy_abs.min()

    for i in range(k):
        x_copy_abs_2 = x_copy_abs.clone()
        noise_iteration_shared = crypten.cryptensor(noise[i], ptype=crypten.mpc.arithmetic)
        x_copy_abs_2 = x_copy_abs_2 + noise_iteration_shared

        max_index = x_copy_abs_2.argmax(one_hot=False)
        max_index_rev = max_index.get_plain_text()
        index_k_argmax[max_index_rev] = 1

        x_copy_abs[max_index_rev] = min_value # set to the maximum value to avoid finding the same index in the next iteration

    return index_k_argmax


# ================== Implemented an updated distributed_noise_and_round from Damgard's paper ==================
@mpc.run_multiprocess(world_size=2)
def distributed_noise_and_round(nabla_input, noise_iterations, noise_output, theta_prev_input, n, d, k=2, a=64, c=0, eta=0.1, t=0):
    nabla = crypten.cryptensor(nabla_input, ptype=crypten.mpc.arithmetic)
    theta_prev = crypten.cryptensor(theta_prev_input, ptype=crypten.mpc.arithmetic)
    noise_output_shared = crypten.cryptensor(noise_output, ptype=crypten.mpc.arithmetic)

    # Local computation: add gradients
    nabla_sum = 0
    for i in range(n):
        nabla_sum += nabla[i]

    # Local computation: aggregate gradients
    nabla_prev = nabla_sum.div(n)

    # Local computaion: perform gradient descend
    theta_t = theta_prev - eta * nabla_prev

    # Ring 2^a ==> TODO if it works, move inside get_min_indices
    # theta_trunc = mod_2a(theta_trunc, a)

    indices = get_indices(theta_t, noise_iterations, k, d, c)

    # make k-sparse
    for i in range(d):
        if indices[i] == 0:
            zero = crypten.cryptensor(0, ptype=crypten.mpc.arithmetic)
            theta_t[i] = zero

    theta_t = theta_t + noise_output_shared

    return theta_t.get_plain_text()
    

def generate_laplace_noise(lmb, ebs, dta, s, d, n_servers):
    full_noise = np.zeros(d)
    for i in range(n_servers):
        s = max(s, 1) # to avoid generating zero noise when the sparsity is 0
        scale = (lmb / ebs) * (2 * np.sqrt(3 * s * np.log(1 / dta)))
        noise = np.random.laplace(0, scale, d)
        full_noise += noise
    return full_noise


def generate_noise(noise_params, T_max, k, p, n_servers=1):
    lmb, ebs, dlt, dp = noise_params
    # TODO: ebs, dlt = ebs / T_max, dlt / T_max

    if dp == 'no_noise':
        return [torch.zeros(p) for _ in range(k)], torch.zeros(p)

    if dp == 'noise_output':
        noise_selection_output = torch.tensor(generate_laplace_noise(lmb, ebs, dlt, k, p, n_servers))
        return [torch.zeros(p) for _ in range(k)], noise_selection_output

    # if noise == 'noise_output_and_per_it'
    noise_selection_iterations = []
    for _ in range(k):
        iteration_noise = torch.tensor(generate_laplace_noise(lmb, ebs, dlt, k, p, n_servers))
        noise_selection_iterations.append(iteration_noise)
    noise_selection_output = torch.tensor(generate_laplace_noise(lmb, ebs, dlt, k, p, n_servers))
    return noise_selection_iterations, noise_selection_output


def solve(data, noise_params, k, T_max, a=64, c=0, eta=0.1, eps=0.001):
    x_input, y_input, theta_star = data # unpack data
    n = len(x_input)
    p = len(x_input[0])

    theta_prev = torch.zeros(p)
    nabla = torch.zeros((n, p))
    nabla = nabla.double()
    theta_prev = theta_prev.double()

    # x_input = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    # theta_prev = torch.tensor([0, 0, 0])
    # y_input = torch.tensor([10, 20, 30])

    iteration_error = [0]
    for t in range(T_max):
        if t % 10 == 0:
            print(t)

        noise_selection_iterations, noise_selection_output = generate_noise(noise_params, T_max, k, p, 1)

        nabla = torch.zeros((n, p))
        nabla = nabla.double()
        theta_prev = theta_prev.double()

        for i in range(n):
            var1 = theta_prev.matmul(x_input[i]) # --> scalar
            var1 = var1 - y_input[i] # --> scalar
            var2 = x_input[i].transpose(0, -1)
            nabla[i] = var2.mul(var1)

        result = distributed_noise_and_round(nabla, noise_selection_iterations, noise_selection_output, theta_prev, n, p, k, a, c, eta, t)
        theta_t = result[0] # just from the first party

        e = torch.norm(theta_t - theta_star, p=2)
        diff = abs(e - iteration_error[-1])

        iteration_error.append(e.item())
        theta_prev = theta_t

        # if diff <= eps:
        #     break

    rel_error = iteration_error[-1] / torch.norm(theta_star, p=2)
    return iteration_error[1:], rel_error

