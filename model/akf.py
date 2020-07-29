#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
from model.vaeakf_v2 import VAEAKF as VAEAKF_v2


class AKF(VAEAKF_v2, nn.Module):
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
        self.estimate_logsigma = torch.nn.parameter.Parameter(torch.randn(state_size)) # noise in processing

        # In linear gauss system, x(k+1) = Ax(k) + Bu(k) + e(k), z(t) = Hx(t) + v(t)
        # in v1, it assumes that A is constant identity matrix
        self.estimate_A = torch.nn.parameter.Parameter(torch.eye(observation_size), requires_grad=False)
        self.estimate_B = torch.nn.parameter.Parameter(torch.randn((state_size, input_size)))
        self.estimate_H = torch.nn.parameter.Parameter(torch.randn((observation_size, state_size)))
        self.estimate_logdelta = torch.nn.parameter.Parameter(torch.randn(observation_size)) # noise in observation

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
            yield mu_prior, torch.log(sigma_prior)
            cov_prior = torch.diag_embed(self.initial_prior_sigma**2)
            for i in range(1, l):
                mu_predict = torch.nn.functional.linear(mu_prior, self.estimate_A) + torch.nn.functional.linear(self.external_input_seq[i-1], self.estimate_B)
                cov_predict = self.estimate_A.matmul(
                   cov_prior
                ).matmul(self.estimate_A.t()) + torch.diag_embed(torch.exp(self.estimate_logsigma)**2)
                K = cov_predict.matmul(self.estimate_H.t()).matmul(
                    (self.estimate_H.matmul(cov_predict).matmul(self.estimate_H.t()) +
                     torch.diag_embed(torch.exp(self.estimate_logdelta)**2)).inverse()
                )
                mu_posteriori = mu_predict + K.matmul(
                    (self.observations_seq[i]-torch.nn.functional.linear(mu_predict, self.estimate_H)).unsqueeze(dim=-1)
                ).squeeze(dim=-1)
                cov_posteriori = (
                        torch.diag_embed(
                            torch.ones((batch_size, self.state_size)).to(self.observations_seq.device) #construct identity matrix
                        ) - K.matmul(self.estimate_H)
                ).matmul(cov_predict)
                yield mu_posteriori, torch.log(torch.sqrt(torch.diagonal(cov_posteriori, dim1=-2,dim2=-1)))
                mu_prior = mu_posteriori
                cov_prior = cov_posteriori

        self.state_mu, self.state_logsigma = [torch.stack(
            tensors, dim=0) for tensors in zip(
            *[x for x in recursive_kalman(self.initial_prior_mu, self.initial_prior_sigma, l)]
        )
        ]



    def sample_state(self, max_prob=True):
        """
        :param max_prob: If True, return the sequence which has the biggest probability. Otherwise, a randomly sampled
        sequence will be returned.
        :return:(l, bs, state_size)
        """
        if self.state_mu is None:
            raise AssertionError('You have to call forward method before sampling state')
        if max_prob:
            return self.state_mu
        noise = torch.randn(self.state_mu.shape).to(self.state_mu.device)
        state = noise*torch.exp(self.state_logsigma) + self.state_mu
        return state

    def multivariateNormal_kl_loss(self, mu1, cov1, mu2, cov2):
        """
        In adaptive kalman filer, the posterior distribution of hidden states is solved theoretically.
        the variational bayes is unnecessary.
        The loss function becomes to the likelihood of observed data.
        More details can be found in Mehra, R. K. (1972). Approaches to Adaptive Filtering. IEEE Transactions on Automatic Control, 17(5), 693–698. https://doi.org/10.1109/TAC.1972.1100100
        """
        return torch.tensor(0).to(mu1.device)
