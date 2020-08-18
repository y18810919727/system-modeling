#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
from model.vaeakf_v2 import VAEAKF as VAEAKF_v2
from common import softplus, inverse_softplus

class AKFVB(VAEAKF_v2, nn.Module):
    """
    Mehra, R. K. (1972). Approaches to Adaptive Filtering. IEEE Transactions on Automatic Control, 17(5), 693–698. https://doi.org/10.1109/TAC.1972.1100100
    """

    def __init__(self,input_size, state_size, observation_size, net_type='lstm', k=16, num_layers=1, L=5):
        """

        :param input_size: The size of external input
        :param state_size:
        :param observation_size:
        :param net_type: the type of VAE
        :param k: size of hidden state
        :param num_layers:
        :param L: The times for sampling auxiliary distribution.
        """

        nn.Module.__init__(self)
        self.k = k
        self.observation_size = observation_size
        self.state_size = state_size
        self.input_size = input_size

        # region Define parameters for training

        # In linear gauss system, x(k+1) = Ax(k) + Bu(k) + e(k), z(t) = Hx(t) + v(t)
        # in v1, it assumes that A is constant identity matrix
        self.estimate_A = torch.nn.parameter.Parameter(torch.eye(observation_size), requires_grad=False)
        self.estimate_B = torch.nn.parameter.Parameter(torch.randn((state_size, input_size)))
        self.estimate_H = torch.nn.parameter.Parameter(torch.randn((observation_size, state_size)))
        self.estimate_logdelta = torch.nn.parameter.Parameter(torch.randn(observation_size)) # noise in observation
        self.estimate_logsigma = torch.nn.parameter.Parameter(torch.randn(state_size)) # noise in processing
        self.estimate_bias = torch.nn.parameter.Parameter(torch.randn(observation_size)) # bias in observation

        # endregion

        self.state_mu = None
        self.state_logsigma = None
        self.external_input_seq = None
        self.observations_seq = None
        self.initial_prior_mu, self.initial_prior_sigma = None, None

    def forward(self, *input):
        external_input_seq, observations_seq, self.initial_prior_mu, self.initial_prior_sigma = input

        self.external_input_seq = external_input_seq
        self.observations_seq = observations_seq

        l, batch_size, _ = external_input_seq.shape

        def recursive_kalman(mu_prior, sigma_prior, l):
            cov_prior = torch.diag_embed(sigma_prior**2)
            for i in range(l):
                # region kalman filter
                mu_predict = torch.nn.functional.linear(mu_prior, self.estimate_A) + torch.nn.functional.linear(self.external_input_seq[i], self.estimate_B)
                cov_predict = self.estimate_A.matmul(
                   cov_prior
                ).matmul(self.estimate_A.t()) + torch.diag_embed(softplus(self.estimate_logsigma)**2)
                K = cov_predict.matmul(self.estimate_H.t()).matmul(
                    (self.estimate_H.matmul(cov_predict).matmul(self.estimate_H.t()) +
                     torch.diag_embed(softplus(self.estimate_logdelta)**2)).inverse()
                )
                mu_posterior = mu_predict + K.matmul(
                    (self.observations_seq[i]-torch.nn.functional.linear(mu_predict, self.estimate_H)).unsqueeze(dim=-1)
                ).squeeze(dim=-1)
                cov_posterior = (
                        torch.diag_embed(
                            torch.ones((batch_size, self.state_size)).to(self.observations_seq.device) #construct identity matrix
                        ) - K.matmul(self.estimate_H)
                ).matmul(cov_predict)
                # endregion
                yield mu_posterior, inverse_softplus(torch.sqrt(torch.diagonal(cov_posterior, dim1=-2,dim2=-1)))
                mu_prior = mu_posterior
                cov_prior = cov_posterior

        self.state_mu, self.state_logsigma = [torch.stack(
            tensors, dim=0) for tensors in zip(
            *[x for x in recursive_kalman(self.initial_prior_mu, self.initial_prior_sigma, l)]
        )
        ]


    def kl_initial(self):
        return torch.tensor(0).to(self.state_mu.device)


