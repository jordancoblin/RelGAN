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

def sparsemax_custom(z):
    # print ("z: ", sess.run(z))
    dim = tf.shape(z)[-1]
    # print ("dim: ", sess.run(dim))
    sorted = tf.sort(z, axis=-1, direction='DESCENDING')
    cumsum = tf.math.cumsum(sorted, axis=-1)
    ind = tf.range(start=1, limit=tf.cast(dim, z.dtype)+1, dtype=z.dtype)
    # print ("ind: ", sess.run(ind))
    # print ("sorted: ", sess.run(sorted))
    # print ("cumsum: ", sess.run(cumsum))
    bound = 1 + ind * sorted
    is_gt = tf.where(tf.math.greater(bound, cumsum), ind*1, ind*0)
    # print ("is_gt: ", sess.run(is_gt))
    
    k = tf.math.reduce_max(is_gt)
    # print ("k: ", sess.run(k))
    tau = (cumsum[tf.cast(k, tf.int32)-1] - 1)/k
    # print ("tau: ", sess.run(tau))
    output = tf.math.maximum(z-tau, tf.cast(0, z.dtype))
    return output

def sparsemax(logits, axis: int = -1) -> tf.Tensor:
    r"""Sparsemax activation function.
    For each batch $i$, and class $j$,
    compute sparsemax activation function:
    $$
    \mathrm{sparsemax}(x)[i, j] = \max(\mathrm{logits}[i, j] - \tau(\mathrm{logits}[i, :]), 0).
    $$
    See [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068).
    Usage:
    >>> x = tf.constant([[-1.0, 0.0, 1.0], [-5.0, 1.0, 2.0]])
    >>> tfa.activations.sparsemax(x)
    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[0., 0., 1.],
           [0., 0., 1.]], dtype=float32)>
    Args:
        logits: A `Tensor`.
        axis: `int`, axis along which the sparsemax operation is applied.
    Returns:
        A `Tensor`, output of sparsemax transformation. Has the same type and
        shape as `logits`.
    Raises:
        ValueError: In case `dim(logits) == 1`.
    """
    logits = tf.convert_to_tensor(logits, name="logits")

    # We need its original shape for shape inference.
    shape = logits.get_shape()
    rank = shape.rank
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if is_last_axis:
        output = _compute_2d_sparsemax(logits)
        output.set_shape(shape)
        return output

    # If dim is not the last dimension, we have to do a transpose so that we can
    # still perform softmax on its last dimension.

    # Swap logits' dimension of dim and its last dimension.
    rank_op = tf.rank(logits)
    axis_norm = axis % rank
    logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))

    # Do the actual softmax on its last dimension.
    output = _compute_2d_sparsemax(logits)
    output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))

    # Make shape inference work since transpose may erase its static shape.
    output.set_shape(shape)
    return output


def _swap_axis(logits, dim_index, last_index, **kwargs):
    return tf.transpose(
        logits,
        tf.concat(
            [
                tf.range(dim_index),
                [last_index],
                tf.range(dim_index + 1, last_index),
                [dim_index],
            ],
            0,
        ),
        **kwargs,
    )


def _compute_2d_sparsemax(logits):
    """Performs the sparsemax operation when axis=-1."""
    shape_op = tf.shape(logits)
    obs = tf.math.reduce_prod(shape_op[:-1])
    dims = shape_op[-1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    # Reshape to [obs, dims] as it is almost free and means the remanining
    # code doesn't need to worry about the rank.
    z = tf.reshape(logits, [obs, dims])

    # sort z
    z_sorted, _ = tf.nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = tf.math.maximum(k_z, 1)
    indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

    # calculate p
    p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))

    # If k_z = 0 or if z = nan, then the input is invalid
    # p_safe = tf.where(
    #     tf.expand_dims(
    #         tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
    #         axis=-1,
    #     ),
    #     tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
    #     p,
    # )

    # Reshape back to original size
    p_safe = tf.reshape(p, shape_op)
    return p_safe

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
# z = [[-1.0, 0.0, 1.0], [-5.0, 1.0, 2.0]]

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