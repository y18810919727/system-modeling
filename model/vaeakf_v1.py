#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn

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
    """ The pre-process layer for MNIST image

    """
    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = self.fc2(t)
        return t


class VAEAKF(nn.Module):
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

        super(VAEAKF, self).__init__()
        self.k = k
        self.observation_size = observation_size
        self.state_size = state_size
        self.input_size = input_size
        self.L = L

        # region Define parameters for training
        self.process_x = PreProcess(input_size + observation_size, k)
        self.net_type = net_type
        if net_type == 'lstm':
            self.rnn = nn.LSTM(k, k, num_layers=num_layers)
        else:
            raise NotImplemented('The rnn %s is not implemented'%net_type)
        self.gauss_decoder = DBlock(k, k*2, state_size)
        self.estimate_logsigma = torch.nn.parameter.Parameter(torch.randn(state_size)) # noise in processing

        # In linear gauss system, x(k+1) = Ax(k) + Bu(k) + e(k), z(t) = Hx(t) + v(t)
        # in v1, it assumes that A is constant identity matrix
        self.estimate_A = torch.nn.parameter.Parameter(torch.eye(observation_size), requires_grad=False)
        self.estimate_B = torch.nn.parameter.Parameter(torch.zeros((state_size, input_size)))
        self.estimate_H = torch.nn.parameter.Parameter(torch.zeros((observation_size, state_size)))
        self.estimate_delta = torch.nn.parameter.Parameter(torch.randn(observation_size)) # noise in observation

        # endregion

        self.mu = None
        self.logsigma = None
        self.external_input_seq = None
        self.observations_seq = None
        self.initial_prior_mu, self.initial_prior_sigma = None, None




    def forward(self, *input):
        external_input_seq, observations_seq, self.initial_prior_mu, self.initial_prior_sigma = input

        self.external_input_seq = external_input_seq
        self.observations_seq = observations_seq

        l, batch_size, _ = external_input_seq.shape
        all_seq = torch.cat([external_input_seq, observations_seq], dim=2)
        all_seq = self.process_x(
            all_seq.contiguous().view(-1, self.observation_size + self.input_size)
        ).view(l, batch_size, self.k)
        z_seq, _ = self.rnn(all_seq)
        self.mu, self.logsigma = self.gauss_decoder(
            z_seq.contiguous().view(-1, self.k)
        )
        self.mu = self.mu.view(l, batch_size, self.state_size)
        self.logsigma = self.logsigma.view(l, batch_size, self.state_size)



    def multivariateNormal_kl_loss(self, mu1, cov1, mu2, cov2):
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
        dist1 = torch.distributions.MultivariateNormal(mu1, torch.diag_embed(cov1))
        dist2 = torch.distributions.MultivariateNormal(mu2, torch.diag_embed(cov2))
        kl = torch.distributions.kl.kl_divergence(dist1, dist2)
        return torch.sum(kl)

    def call_loss(self):

        l, bs, _  = self.observations_seq.shape

        # region calculating kl divergence
        prior_mu = torch.zeros((l, bs, self.state_size)).to(self.observations_seq.device)
        prior_cov = torch.zeros((l, bs, self.state_size)).to(self.observations_seq.device)

        prior_mu[0]=self.initial_prior_mu
        prior_mu[1:] = torch.nn.functional.linear(
            self.mu[0:-1],self.estimate_A) + torch.nn.functional.linear(self.external_input_seq[:-1], self.estimate_B)
        prior_cov[0] = self.initial_prior_sigma**2
        prior_cov[1:] = torch.nn.functional.linear(
            torch.exp(self.logsigma[:-1])**2, self.estimate_A
        ) + torch.exp(self.estimate_logsigma)**2

        kl = self.multivariateNormal_kl_loss(prior_mu, prior_cov, self.mu, torch.exp(self.logsigma)**2)
        # endregion

        # region calculation generative probability p(x|z)
        def estimate_generative_probability():
            state = self.sample_state(max_prob=False)
            observation_mu = torch.nn.functional.linear(state, self.estimate_H)
            observation_cov = torch.diag_embed(
                torch.exp(self.estimate_delta)**2,
            )
            observation_normal_dist = torch.distributions.MultivariateNormal(observation_mu, observation_cov)
            return torch.sum(observation_normal_dist.log_prob(self.observations_seq))

        generative_likelihood = sum([estimate_generative_probability() for _ in range(self.L)])/self.L
        # endregion

        # maximun -kl + generative_likelihood
        return (kl - generative_likelihood)/bs

    def sample_state(self, max_prob=True):
        """
        :param max_prob: If True, return the sequence which has the biggest probability. Otherwise, a randomly sampled
        sequence will be returned.
        :return:(l, bs, state_size)
        """
        if self.mu is None:
            raise AssertionError('You have to call forward method before sampling state')
        if max_prob:
            return self.mu
        noise = torch.randn(self.mu.shape).to(self.mu.device)
        state = noise*torch.exp(self.logsigma) + self.mu
        return state
