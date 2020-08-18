#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
from common import softplus

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
        self.num_layers = num_layers

        # region Define parameters for training
        self.process_x = PreProcess(input_size + observation_size, k)
        self.net_type = net_type
        if net_type == 'lstm':
            self.rnn = nn.LSTM(k, k, num_layers=num_layers)
        else:
            raise NotImplemented('The rnn %s is not implemented'%net_type)
        self.gauss_decoder = DBlock(k, k*2, state_size)

        # It encodes initial distribution of state and feed it to lstm as h0
        self.dist_encoder = nn.Sequential(
            torch.nn.Linear(state_size*2, k),
            torch.nn.Tanh(),
            torch.nn.Linear(k, k),
            torch.nn.Tanh()
        )

        # In linear gauss system, x(k+1) = Ax(k) + Bu(k) + e(k), z(t) = Hx(t) + v(t)
        # in v1, it assumes that A is constant identity matrix
        self.estimate_A = torch.nn.parameter.Parameter(torch.eye(observation_size), requires_grad=False)
        self.estimate_B = torch.nn.parameter.Parameter(torch.zeros((state_size, input_size)))
        self.estimate_H = torch.nn.parameter.Parameter(torch.zeros((observation_size, state_size)))
        self.estimate_logsigma = torch.nn.parameter.Parameter(torch.randn(state_size)) # noise in processing
        self.estimate_logdelta = torch.nn.parameter.Parameter(torch.randn(observation_size)) # noise in observation
        self.estimate_bias = torch.nn.parameter.Parameter(torch.randn(observation_size)) # bias in observation

        # endregion

        self.state_mu = None
        self.state_logsigma = None
        self.external_input_seq = None
        self.observations_seq = None
        self.initial_prior_mu, self.initial_prior_sigma = None, None
        self.initial_hidden_state = None




    def forward(self, *input):
        external_input_seq, observations_seq, self.initial_prior_mu, self.initial_prior_sigma = input


        self.external_input_seq = external_input_seq
        self.observations_seq = observations_seq

        l, batch_size, _ = external_input_seq.shape

        self.initial_hidden_state = self.dist_encoder(
            torch.cat([self.initial_prior_mu, self.initial_prior_sigma], dim=1)
        ).repeat(self.num_layers, 1, 1)
        all_seq = torch.cat([external_input_seq, observations_seq], dim=2)
        all_seq = self.process_x(
            all_seq.contiguous().view(-1, self.observation_size + self.input_size)
        ).view(l, batch_size, self.k)
        z_seq, _ = self.rnn(all_seq, (self.initial_hidden_state, torch.zeros_like(self.initial_hidden_state)))
        self.state_mu, self.state_logsigma = self.gauss_decoder(
            z_seq.contiguous().view(-1, self.k)
        )
        self.state_mu = self.state_mu.view(l, batch_size, self.state_size)
        self.state_logsigma = self.state_logsigma.view(l, batch_size, self.state_size)



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
        dist1 = torch.distributions.MultivariateNormal(mu1, cov1)
        dist2 = torch.distributions.MultivariateNormal(mu2, cov2)
        kl = torch.distributions.kl.kl_divergence(dist1, dist2)
        return torch.sum(kl)

    def call_loss(self):

        l, bs, _  = self.observations_seq.shape

        # region calculating kl divergence

        # Linear transformation of gaussian dist is also a distribution. Ax+b \sim N(AE(x)+b, ACov(x)A^T)
        prior_mu = torch.cat([self.initial_prior_mu.unsqueeze(dim=0), self.state_mu[:-1]], dim=0)
        prior_mu = torch.nn.functional.linear(
            prior_mu, self.estimate_A) + torch.nn.functional.linear(self.external_input_seq, self.estimate_B)

        prior_cov = torch.diag_embed(
            torch.cat([self.initial_prior_sigma.unsqueeze(dim=0)**2, softplus(self.state_logsigma[:-1])**2], dim=0)
        )
        prior_cov = self.estimate_A.matmul(prior_cov).matmul(self.estimate_A.t()) + torch.diag_embed(softplus(self.estimate_logsigma)**2)

        kl = self.multivariateNormal_kl_loss(prior_mu, prior_cov, self.state_mu, torch.diag_embed(softplus(self.state_logsigma)**2))
        # endregion

        # region calculation generative probability p(x|z)
        generative_likelihood = self.estimate_generative_logprobability()
        # endregion


        # Minimize reconstruction error of dist_encoder
        kl_initial = self.kl_initial()

        # maximun -kl + generative_likelihood
        return (kl - generative_likelihood + kl_initial)/bs

    def estimate_generative_logprobability(self):

        def estimate_generative_logprobability_from_sample():
            state = self.sample_state(max_prob=False)
            observation_mu = torch.nn.functional.linear(state, self.estimate_H) + self.estimate_bias
            observation_cov = torch.diag_embed(
                softplus(self.estimate_logdelta)**2,
                )
            observation_normal_dist = torch.distributions.MultivariateNormal(observation_mu, observation_cov)
            return torch.sum(observation_normal_dist.log_prob(self.observations_seq))

        generative_likelihood = sum([estimate_generative_logprobability_from_sample() for _ in range(self.L)])/self.L
        return generative_likelihood

    def kl_initial(self):

        estimate_begin_mu, estimate_begin_logsigma = self.gauss_decoder(self.initial_hidden_state[0])
        kl_initial = self.multivariateNormal_kl_loss(self.initial_prior_mu,
                                                     torch.diag_embed(self.initial_prior_sigma**2),
                                                     estimate_begin_mu,
                                                     torch.diag_embed(softplus(estimate_begin_logsigma)**2))
        return kl_initial

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
        noise = torch.randn_like(self.state_mu)
        state = noise*softplus(self.state_logsigma) + self.state_mu
        return state

    def sigma_interval(self, e):
        return self.state_mu - e * softplus(self.state_logsigma), \
               self.state_mu + e * softplus(self.state_logsigma)
