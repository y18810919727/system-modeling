#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import torch
from torch import nn
from common import softplus, inverse_softplus, cov2logsigma, logsigma2cov
from model.func import weighted_linear, normal_differential_sample


class DBlock(nn.Module):
    """ A basie building block for parametralize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma


class PreProcess(nn.Module):

    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = self.fc2(t)
        return t


class MLP(nn.Module):
    def __init__(self, input_size, hidden_num, out_size, num_mlp_layers, nonlinearity=torch.nn.Tanh):
        super(MLP, self).__init__()
        L = nn.ModuleList()
        L.append(nn.Linear(input_size, hidden_num))
        L.append(nonlinearity())
        for _ in range(num_mlp_layers-1):
            L.append(nn.Linear(hidden_num, hidden_num))
            L.append(nonlinearity())
        L.append(nn.Linear(hidden_num, out_size))
        self.mlp = nn.Sequential(
            *L
        )

    def forward(self, x):
        return self.mlp(x)

from torch.distributions.multivariate_normal import MultivariateNormal, _precision_to_scale_tril, _batch_mahalanobis


class DiagMultivariateNormal(torch.distributions.multivariate_normal.MultivariateNormal):
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) != 1:
            raise ValueError("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.")

        loc_ = loc.unsqueeze(-1)  # temporarily add dim on right
        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.scale_tril, loc_ = torch.broadcast_tensors(scale_tril, loc_)
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.covariance_matrix, loc_ = torch.broadcast_tensors(covariance_matrix, loc_)
        else:
            if precision_matrix.dim() < 2:
                raise ValueError("precision_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.precision_matrix, loc_ = torch.broadcast_tensors(precision_matrix, loc_)
        self.loc = loc_[..., 0]  # drop rightmost dim

        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super(MultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            #self._unbroadcasted_scale_tril = torch.cholesky(covariance_matrix)
            self._unbroadcasted_scale_tril = torch.sqrt(covariance_matrix)
        else:  # precision_matrix is not None
            raise NotImplementedError('Only covariance_matrix or scale_tril may be specified')



