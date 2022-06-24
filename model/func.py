#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

def weighted_linear(input, linears, weight):
    """

    Args:
        input: len, bs, k
        linears: nn.ModuleList [ Linear(k,out) for _ in num_linears ]
        weight:

    Returns:

    """
    num_linears = len(linears)
    assert num_linears == weight.shape[-1]
    weight_weighted = torch.sum(
        torch.stack([linear.weight for linear in linears], dim=0) * weight.unsqueeze(dim=-1).unsqueeze(dim=-1),
        dim=-3
    )
    bias_weighted = torch.sum(
        torch.stack([linear.bias for linear in linears], dim=0) * weight.unsqueeze(dim=-1),
        dim=-2) if linears[0].bias is not None else 0
    return (weight_weighted @ input.unsqueeze(dim=-1)).squeeze(dim=-1) + bias_weighted

def normal_differential_sample(normal_dist, n=1):
    """

    Args:
        normal_dist: torch.distributions.MultivariateNormal
        n: The number for sampling

    Returns:tensor with shape [n, *with mu()mu.shape]
    """
    noise = torch.randn((n, *normal_dist.loc.size()), dtype=normal_dist.loc.dtype,
                        device=normal_dist.loc.device, layout=normal_dist.loc.layout)
    # n_samples = (torch.cholesky(normal_dist.covariance_matrix) @ noise.unsqueeze(dim=-1)).squeeze(dim=-1) + normal_dist.loc
    # It seems that torch.cholesky has a strange bug during decomposing batch matrix when using GPU
    # https://discuss.pytorch.org/t/cuda-illegal-memory-access-when-using-batched-torch-cholesky/51624/3

    # In our experiments, all of the covariance matrix are diagonal matrix. So
    n_samples = (torch.sqrt(normal_dist.covariance_matrix) @ noise.unsqueeze(dim=-1)).squeeze(dim=-1) + normal_dist.loc
    if n==1:
        n_samples = n_samples.squeeze(dim=0)
    return n_samples


def multivariate_normal_kl_loss(mu1, cov1, mu2, cov2):
    """
    Calculating the kl divergence of two  Multivariate Normal distributions
    references:
    1. https://pytorch.org/docs/stable/distributions.html?highlight=kl#torch.distributions.kl.kl_divergence
    2. https://zhuanlan.zhihu.com/p/22464760
    :param mu1: (Len, batch_size, k)
    :param mu2:
    :param sigma1:
    :param sigma2:
    :return:  a scalar loss
    """
    from model.common import DiagMultivariateNormal as MultivariateNormal
    dist1 = MultivariateNormal(mu1, cov1)
    dist2 = MultivariateNormal(mu2, cov2)
    kl = torch.distributions.kl.kl_divergence(dist1, dist2)
    return torch.sum(kl)


def kld_gauss(mu_q, logvar_q, mu_p, logvar_p, sum=True):
    # Goal: Minimize KL divergence between q_pi(z|xi) || p(z|xi)
    # This is equivalent to maximizing the ELBO: - D_KL(q_phi(z|xi) || p(z)) + Reconstruction term
    # This is equivalent to minimizing D_KL(q_phi(z|xi) || p(z))
    term1 = logvar_p - logvar_q - 1
    term2 = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
    if sum:
        kld = 0.5 * torch.sum(term1 + term2)
    else:
        kld = 0.5 * (term1 + term2)

    return kld


def zeros_like_with_shape(tensor, shape=None):
    if shape is None:
        shape = tensor.shape

    return torch.zeros(shape,
                       device=tensor.device,
                       dtype=tensor.dtype,
                       layout=tensor.layout
                       )
