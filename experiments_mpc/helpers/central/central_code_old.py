import crypten
import torch

crypten.init()
torch.set_num_threads(1)

# Sorting vs. selecting the min k times
# sorting: O(nlogn); index = np.argsort(v_abs)
# selecting the min k times: O(nk); implementation below
# sorting is too slow, especially when k << n

# k - how many elements to make nonzero => make (n - k) zero
def truncation(v, k):
    n = len(v)
    v_abs = torch.abs(v)
    max_value = torch.max(v_abs)
    indices_min = []

    # save the index of the smallest (n - k) values
    k = n - k
    for i in range(k):
        index_min = torch.argmin(v_abs)

        indices_min.append(index_min)
        v_abs[index_min] = max_value # set to the maximum value to avoid finding it again

    # set the smallest k values to zero
    for i in indices_min:
        v[i] = 0

    return v


def iht_central(data, noise, k, T_max, eta=0.1, eps=0.000001,):

    x, y, theta_star = data # unpack data
    n = len(x)
    p = len(x[0])

    theta_prev = torch.zeros(p)
  
    # x = torch.tensor([[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]]) # n = 3, p = 4
    # theta_star = torch.tensor([1, 2, 3, 4])
    # noise = torch.tensor([1, 1, 1])
    # y = x @ theta_star + noise

    # Line 3: users computes their gradient locally
    iteration_error = [0]
    for t in range(T_max):
        all_gradients = []

        theta_prev = theta_prev.double()
        for i in range(n):
            var1 = theta_prev.dot(x[i])
            var1 = var1 - y[i]
            var2 = x[i].t()
            nabla_i = var2 * var1
            all_gradients.append(nabla_i)

        # Line 4: aggregate gradients and noise
        nabla_prev = torch.stack(all_gradients).sum(axis=0)
        nabla_prev = nabla_prev + noise
        nabla_prev = nabla_prev /  len(all_gradients)

        # Line 5: perform gradient descend
        theta_t = theta_prev - eta * nabla_prev

        # Line 6: truncation - make k-sparse
        if k > 0 and k < p:
            theta_t = truncation(theta_t, k)

        e = torch.norm(theta_t - theta_star, p=2)
        diff = abs(e - iteration_error[-1])

        iteration_error.append(e.item())
        theta_prev = theta_t

        if diff <= eps:
            break

    rel_error = iteration_error[-1] / torch.norm(theta_star, p=2)
    return iteration_error[1:], rel_error