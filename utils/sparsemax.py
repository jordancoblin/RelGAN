import numpy as np
import torch
# import tensorflow_addons as tfa
import tensorflow as tf

# TODO: implement backward algo
# TODO: figure out how to incorporate this into TF
# def sparsemax(z):
#     sorted = np.sort(z)[::-1]
#     cumsum = np.cumsum(sorted)
#     ind = np.arange(start=1, stop=len(z)+1)
#     bound = 1 + ind * sorted
#     is_gt = np.greater(bound, cumsum)
#     k = np.max(is_gt * ind)
#     tau = (cumsum[k-1] - 1)/k
#     output = np.clip(z-tau, a_min=0, a_max=None)
#     return output

def sparsemax(z):
    print ("z: ", sess.run(z))
    dim = tf.shape(z)[-1]
    print ("dim: ", sess.run(dim))
    sorted = tf.sort(z, axis=-1, direction='DESCENDING')
    cumsum = tf.math.cumsum(sorted, axis=-1)
    ind = tf.range(start=1, limit=tf.cast(dim, z.dtype)+1, dtype=z.dtype)
    print ("ind: ", sess.run(ind))
    print ("sorted: ", sess.run(sorted))
    print ("cumsum: ", sess.run(cumsum))
    bound = 1 + ind * sorted
    is_gt = tf.where(tf.math.greater(bound, cumsum), ind*1, ind*0)
    print ("is_gt: ", sess.run(is_gt))
    
    k = tf.math.reduce_max(is_gt)
    print ("k: ", sess.run(k))
    tau = (cumsum[tf.cast(k, tf.int32)-1] - 1)/k
    print ("tau: ", sess.run(tau))
    output = tf.math.maximum(z-tau, tf.cast(0, z.dtype))
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

def project_simplex_grad(dout, w_star):
    supp = w_star > 0
    masked = dout.masked_select(supp)
    nnz = supp.to(dtype=dout.dtype).sum()
    masked -= masked.sum() / nnz
    out = dout.new(dout.size()).zero_()
    out[supp] = masked
    return(out)

def grad_sparsemax(op, grad):
    spm = op.outputs[0]
    support = tf.cast(spm > 0, spm.dtype)

    # Calculate \hat{v}, which will be a vector (scalar for each z)
    v_hat = tf.reduce_sum(tf.mul(grad, support), 1) / tf.reduce_sum(support, 1)

    # Calculates J(z) * v
    return [support * (grad - v_hat[:, np.newaxis])]

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

with tf.compat.v1.Session() as sess: 
    s = sparsemax(tf.constant(z))
    print(s.eval())

print("## Project Simplex")
s2 = project_simplex(torch.Tensor(z))
print(s2)

print("## Yuxin Sparsemax")
s3 = sparsemax_yuxin(torch.Tensor(z))
print(s3)

# x = tf.constant([[-1.0, 0.0, 1.0], [-5.0, 1.0, 2.0]])
# sparse_x = tfa.activations.sparsemax(tf.constant(z))
# print(sparse_x)