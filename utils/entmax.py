"""
Bisection implementation of alpha-entmax (Peters et al., 2019).
Backward pass wrt alpha per (Correia et al., 2019). See
https://arxiv.org/pdf/1905.05702 for detailed description.
"""
# Author: Goncalo M Correia
# Author: Ben Peters
# Author: Vlad Niculae <vlad@vene.ro>

# import torch
# import torch.nn as nn
# from torch.autograd import Function

import sparsemax
import tensorflow as tf

# Potential plan... try porting bisect to TF. Check gradient using custom gradient defined here: https://gist.github.com/BenjaminWegener/8fad40ffd80fbe9087d13ad464a48ca9
# Compare forward and backward results with torch implementation
# Add unit tests

def _gp(x, alpha):
        return x ** (alpha - 1)

def _gp_inv(y, alpha):
    return y ** (1 / (alpha - 1))

def _p(X, alpha):
    return _gp_inv(tf.clip_by_value(X, clip_value_min=0, clip_value_max=tf.float32.max), alpha)

def entmax_bisect_tf(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
    if not isinstance(alpha, tf.Tensor):
        alpha = tf.constant(alpha, dtype=X.dtype)
    
    alpha_shape = list(X.shape)
    alpha_shape[dim] = 1
    # print("alpha1: ", sess.run(alpha))
    alpha = tf.broadcast_to(alpha, alpha_shape)
    # print("alpha: ", sess.run(alpha))

    d = X.shape[dim]

    max_val = tf.reduce_max(X, axis=dim, keepdims=True)
    X = X * (alpha - 1)
    max_val = max_val * (alpha - 1)
    # print("max_val: ", sess.run(max_val))

    # # Note: when alpha < 1, tau_lo > tau_hi. This still works since dm < 0.
    tau_lo = max_val - _gp(1, alpha)
    tau_hi = max_val - _gp(1 / int(d), alpha)

    f_lo = tf.reduce_sum(_p(X - tau_lo, alpha), axis=dim) - 1
    # print("f_lo: ", sess.run(f_lo))

    dm = tau_hi - tau_lo
    # print("dm: ", sess.run(dm))

    for it in range(n_iter):

        dm /= 2
        tau_m = tau_lo + dm
        p_m = _p(X - tau_m, alpha)
        f_m = tf.reduce_sum(p_m, axis=dim) - 1

        f_mask = f_m * f_lo >= 0
        mask = tf.expand_dims(f_mask, dim)
        # mask = (f_m * f_lo >= 0).unsqueeze(dim)
        tau_lo = tf.where(mask, tau_m, tau_lo)

    if ensure_sum_one:
        # p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)
        p_m /= tf.expand_dims(tf.reduce_sum(p_m, axis=dim), axis=dim)

    # # ctx.save_for_backward(p_m)

    # return p_m
    return p_m

@tf.custom_gradient
def entmax_bisect_tf_custom_grad(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
    outputs = entmax_bisect_tf(X, alpha, dim, n_iter, ensure_sum_one)

    def grad_fn(d_outputs):
        with tf.name_scope('entmax_grad'):
            outputs_sqrt = tf.math.sqrt(outputs)
            d_inputs = d_outputs * outputs_sqrt
            q = tf.reduce_sum(d_inputs, axis=dim, keep_dims=True) 
            q = q / tf.reduce_sum(outputs_sqrt, axis=dim, keep_dims=True)
            d_inputs -= q * outputs_sqrt
            return d_inputs
    
    return outputs, grad_fn

# Taken from https://github.com/deep-spin/entmax/blob/master/entmax/root_finding.py
# Ported this over to tensorflow, but keeping the original for debugging purposes.
# class EntmaxBisectFunction(Function):
#     @classmethod
#     def _gp(cls, x, alpha):
#         return x ** (alpha - 1)

#     @classmethod
#     def _gp_inv(cls, y, alpha):
#         return y ** (1 / (alpha - 1))

#     @classmethod
#     def _p(cls, X, alpha):
#         return cls._gp_inv(torch.clamp(X, min=0), alpha)

#     @classmethod
#     def forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):

#         if not isinstance(alpha, torch.Tensor):
#             alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)

#         alpha_shape = list(X.shape)
#         alpha_shape[dim] = 1
#         alpha = alpha.expand(*alpha_shape)
#         print("alpha orig: ", alpha)

#         ctx.alpha = alpha
#         ctx.dim = dim
#         d = X.shape[dim]

#         max_val, _ = X.max(dim=dim, keepdim=True)
#         X = X * (alpha - 1)
#         max_val = max_val * (alpha - 1)
#         print("max_val_orig: ", max_val)

#         # Note: when alpha < 1, tau_lo > tau_hi. This still works since dm < 0.
#         tau_lo = max_val - cls._gp(1, alpha)
#         tau_hi = max_val - cls._gp(1 / d, alpha)

#         f_lo = cls._p(X - tau_lo, alpha).sum(dim) - 1
#         print("f_lo_orig: ", f_lo)

#         dm = tau_hi - tau_lo
#         print("dm: ", dm)

#         for it in range(n_iter):

#             dm /= 2
#             tau_m = tau_lo + dm
#             p_m = cls._p(X - tau_m, alpha)
#             f_m = p_m.sum(dim) - 1

#             mask = (f_m * f_lo >= 0).unsqueeze(dim)
#             tau_lo = torch.where(mask, tau_m, tau_lo)

#         if ensure_sum_one:
#             p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)

#         ctx.save_for_backward(p_m)

#         return p_m

#     @classmethod
#     def backward(cls, ctx, dY):
#         Y, = ctx.saved_tensors

#         gppr = torch.where(Y > 0, Y ** (2 - ctx.alpha), Y.new_zeros(1))

#         dX = dY * gppr
#         q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
#         q = q.unsqueeze(ctx.dim)
#         dX -= q * gppr

#         d_alpha = None
#         if ctx.needs_input_grad[1]:

#             # alpha gradient computation
#             # d_alpha = (partial_y / partial_alpha) * dY
#             # NOTE: ensure alpha is not close to 1
#             # since there is an indetermination
#             # batch_size, _ = dY.shape

#             # shannon terms
#             S = torch.where(Y > 0, Y * torch.log(Y), Y.new_zeros(1))
#             # shannon entropy
#             ent = S.sum(ctx.dim).unsqueeze(ctx.dim)
#             Y_skewed = gppr / gppr.sum(ctx.dim).unsqueeze(ctx.dim)

#             d_alpha = dY * (Y - Y_skewed) / ((ctx.alpha - 1) ** 2)
#             d_alpha -= dY * (S - Y_skewed * ent) / (ctx.alpha - 1)
#             d_alpha = d_alpha.sum(ctx.dim).unsqueeze(ctx.dim)

#         return dX, d_alpha, None, None, None


# def entmax_bisect(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
#     """alpha-entmax: normalizing sparse transform (a la softmax).
#     Solves the optimization problem:
#         max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.
#     where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,
#     using a bisection (root finding, binary search) algorithm.
#     This function is differentiable with respect to both X and alpha.
#     Parameters
#     ----------
#     X : torch.Tensor
#         The input tensor.
#     alpha : float or torch.Tensor
#         Tensor of alpha parameters (> 1) to use. If scalar
#         or python float, the same value is used for all rows, otherwise,
#         it must have shape (or be expandable to)
#         alpha.shape[j] == (X.shape[j] if j != dim else 1)
#         A value of alpha=2 corresponds to sparsemax, and alpha=1 would in theory recover
#         softmax. For numeric reasons, this algorithm does not work with `alpha=1`: if you
#         want softmax, we recommend `torch.nn.softmax`.
#     dim : int
#         The dimension along which to apply alpha-entmax.
#     n_iter : int
#         Number of bisection iterations. For float32, 24 iterations should
#         suffice for machine precision.
#     ensure_sum_one : bool,
#         Whether to divide the result by its sum. If false, the result might
#         sum to close but not exactly 1, which might cause downstream problems.
#     Returns
#     -------
#     P : torch tensor, same shape as X
#         The projection result, such that P.sum(dim=dim) == 1 elementwise.
#     """
#     return EntmaxBisectFunction.apply(X, alpha, dim, n_iter, ensure_sum_one)

# Test softmax recovery
# z = torch.Tensor([2.5, 0.2, 0.1, 3, 0.1, 2.5])
# e = entmax_bisect(z, alpha=1+1e-4)
# print("sparse: ", e)

# z = torch.Tensor([2.5, 0.2, 0.1, 3, 0.1, 2.5])
# s = torch.nn.Softmax(dim=-1)(z)
# print("soft: ", s)

# z = [2.5, 0.2, 0.1, 3, 0.1, 2.5, 2.0, 1.0]
# z = [[2.5, 0.2, 0.1, 3],[5.0, 4.5, 1.5, 0.5]]
# e = entmax_bisect(torch.Tensor(z), alpha=1.3)
# print("entmax_orig: ", e)

# with tf.compat.v1.Session() as sess: 
#     e2 = entmax_bisect_tf(tf.constant(z), alpha=1.3, sess=sess)
#     print("entmax: ", sess.run(e2))

# s = torch.nn.Softmax(dim=-1)(torch.Tensor(z))
# print("soft: ", s)

# with tf.compat.v1.Session() as sess: 
#     s = sparsemax.sparsemax(tf.constant(z))
#     print("sparse: ", sess.run(s))