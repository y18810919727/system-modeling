#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn

from model.vaeakf_v1 import VAEAKF as VAEAKF_v1
from model.vaeakf_v1 import *
from common import softplus


class ObservationModel(nn.Module):
    def __init__(self, state_size, observation_size, k=16, num_layers=1):
        """
        Args:
            state_size:
            observation_size:
        """
        super(ObservationModel, self).__init__()
        self.process_z = PreProcess(state_size, k)
        self.rnn = nn.LSTM(k, k, num_layers=num_layers)
        self.gauss_decoder = DBlock(k, k*2, observation_size)

    def forward(self, state_seq, hn=None):

        l, batch_size, _ = state_seq.shape
        feat_seq = self.process_z(state_seq)
        z_seq, _ = self.rnn(feat_seq, hn)
        observation_mu, observation_logsigma = self.gauss_decoder(z_seq)
        return observation_mu, observation_logsigma


class VAEAKFFixb(VAEAKF_v1, nn.Module):
    def __init__(self, input_size, state_size, observation_size, net_type='lstm', k=16, num_layers=1, L=5):
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
        self.L = L
        self.num_layers = num_layers

        # region Define parameters for training
        self.process_x = PreProcess(input_size + observation_size, k)
        self.net_type = net_type
        if net_type == 'lstm':
            self.rnn = nn.LSTM(k, k, num_layers=num_layers)
            self.rnn_gene = nn.LSTM(k, k, num_layers=num_layers)
        else:
            raise NotImplemented('The rnn %s is not implemented'%net_type)
        self.gauss_decoder = DBlock(k, k*2, state_size)
        self.observation_model = ObservationModel(state_size, observation_size)

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
        #self.estimate_B = torch.nn.parameter.Parameter(torch.zeros((state_size, input_size)))
        #  in v5, it assumes that B is known.
        self.estimate_B = torch.nn.parameter.Parameter(
            torch.FloatTensor(
                [[1.21, 0.239, 1.051]]
            ), requires_grad=False
        )
        self.estimate_logsigma = torch.nn.parameter.Parameter(torch.randn(state_size)) # noise in processing
        self.estimate_bias = torch.nn.parameter.Parameter(torch.randn(observation_size)) # bias in observation

        # endregion

        self.state_mu = None
        self.state_logsigma = None
        self.external_input_seq = None
        self.observations_seq = None
        self.initial_prior_mu, self.initial_prior_sigma = None, None
        self.initial_hidden_state = None

    def estimate_generative_logprobability(self):

        def estimate_generative_logprobability_from_sample():
            state = self.sample_state(max_prob=False)
            observation_mu, observation_logsigma = self.observation_model(state)

            observation_cov = torch.diag_embed(
                    softplus(observation_logsigma)**2,
                )
            observation_normal_dist = torch.distributions.MultivariateNormal(observation_mu, observation_cov)
            return torch.sum(observation_normal_dist.log_prob(self.observations_seq))

        generative_likelihood = sum([estimate_generative_logprobability_from_sample() for _ in range(self.L)])/self.L
        return generative_likelihood

