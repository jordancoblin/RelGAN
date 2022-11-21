import numpy as np
import torch

def sparsemax(z):
    sorted = np.sort(z)[::-1]
    cumsum = np.cumsum(sorted)
    ind = np.arange(start=1, stop=len(z)+1)
    bound = 1 + ind * sorted
    is_gt = np.greater(bound, cumsum)
    k = np.max(is_gt * ind)
    tau = (cumsum[k-1] - 1)/k
    output = np.clip(z-tau, a_min=0, a_max=None)
    return output

def project_simplex(v, z=1):
    v_sorted, _ = torch.sort(v, dim=0, descending=True)
    cssv = torch.cumsum(v_sorted, dim=0) - z
    ind = torch.arange(1, 1 + len(v)).to(dtype=v.dtype)
    cond = v_sorted - cssv / ind > 0
    rho = ind.masked_select(cond)[-1]
    tau = cssv.masked_select(cond)[-1] / rho
    w = torch.clamp(v - tau, min=0)
    return w

def sparsemax_yuxin(input, dim=-1):
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        # input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        # range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=zs.device, dtype=input.dtype).view(1, -1)
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=zs.device, dtype=input.dtype)
        # range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[1] + 1

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        print("tau: ", taus)
        taus = taus.expand_as(input)

        # Sparsemax
        output = torch.max(torch.zeros_like(input), input - taus)

        return output


print("## Sparsemax")
z = [2.5, 0.2, 0.1, 3, 0.1, 2.5]
s = sparsemax(z)
print(s)

print("## Project Simplex")
s2 = project_simplex(torch.Tensor(z))
print(s2)

print("## Yuxin Sparsemax")
s3 = sparsemax_yuxin(torch.Tensor(z))
print(s3)