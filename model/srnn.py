#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

from torch import nn
from model.common import DBlock, PreProcess, MLP

from common import softplus, inverse_softplus, cov2logsigma, logsigma2cov, split_first_dim, merge_first_two_dims
from model.func import normal_differential_sample, multivariate_normal_kl_loss
from model.common import DiagMultivariateNormal as MultivariateNormal

class SRNN(nn.Module):
    def __init__(self, input_size, state_size, observations_size, net_type='rnn', k=16, num_layers=1,
                 filtering=True, D=1):

        super(SRNN, self).__init__()

        self.k = k
        self.observations_size = observations_size
        self.state_size = state_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.filtering = filtering
        self.D = D
        if net_type == 'rnn':
            RnnClass = torch.nn.RNN
        elif net_type == 'gru':
            RnnClass = torch.nn.GRU
        else:
            raise NotImplementedError('The posterior rnn with type %d is not implemented!' % net_type)

        self.input_rnn = RnnClass(k, k, num_layers)
        self.process_u = PreProcess(input_size, k)
        self.process_x = PreProcess(observations_size, k)
        if filtering:
            self.posterior_a_layer = MLP(2*k, k, k, 1)
        else:
            self.posterior_a_layer = RnnClass(2*k, k, num_layers)
        self.posterior_gauss = DBlock(state_size+k, 2*k, state_size)

        self.prior_gauss = DBlock(state_size+k, 2*k, state_size)
        self.decoder = DBlock(state_size+k, 2*k, observations_size)

        self.state_mu = None
        self.state_logsigma = None
        self.external_input_seq = None
        self.observations_seq = None
        self.initial_prior_mu, self.initial_prior_logsigma = None, None
        self.sampled_state = None
        self.weight_initial_hidden_state = None
        self.d_seq = None
        self.a_seq = None
        self.external_input_seq_embed = None
        self.observations_seq_embed = None
        self.memory_state = None

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        self.memory_state = memory_state
        d0 = None if memory_state is None else memory_state['dn']
        self.external_input_seq = external_input_seq
        self.observations_seq = observations_seq

        l, batch_size, _ = external_input_seq.size()

        self.external_input_seq_embed = self.process_u(self.external_input_seq)
        self.d_seq, dn = self.input_rnn(self.external_input_seq_embed, d0)

        self.observations_seq_embed = self.process_x(self.observations_seq)
        self.a_seq = self.posterior_a_layer(
            torch.cat([self.d_seq, self.observations_seq_embed], dim=2).flip((0,))
        ).flip((0,))
        if not self.filtering:
            self.a_seq = self.a_seq[0]

        z_t_minus_one = torch.zeros((batch_size, self.state_size), device=self.external_input_seq.device) if memory_state is None else memory_state['zn']
        self.state_mu = []
        self.state_logsigma = []
        self.sampled_state = []
        for t in range(l):
            z_t_mean_res, z_t_logsigma = self.posterior_gauss(
                torch.cat([z_t_minus_one, self.a_seq[t]], dim=-1)
            )
            z_t_mean = self.prior_gauss(
                torch.cat([z_t_minus_one, self.d_seq[t]], dim=-1)
            )[0] + z_t_mean_res

            z_t_minus_one = normal_differential_sample(
                MultivariateNormal(z_t_mean, logsigma2cov(z_t_logsigma))
            )

            self.state_mu.append(z_t_mean)
            self.state_logsigma.append(z_t_logsigma)
            self.sampled_state.append(z_t_minus_one)

        self.state_mu = torch.stack(self.state_mu, dim=0)
        self.state_logsigma = torch.stack(self.state_logsigma, dim=0)
        self.sampled_state = torch.stack(self.sampled_state, dim=0)

        return self.state_mu, self.state_logsigma, {'dn': dn, "zn": z_t_minus_one}

    def forward_prediction(self, external_input_seq, max_prob=False, memory_state=None):
        self.memory_state = memory_state

        l, batch_size, _ = external_input_seq.size()
        d0 = None if memory_state is None else memory_state['dn']
        z_t_minus_one = torch.zeros((batch_size, self.state_size), device=self.external_input_seq.device) if memory_state is None else memory_state['zn']
        self.external_input_seq = external_input_seq
        self.external_input_seq_embed = self.process_u(self.external_input_seq)
        self.d_seq, dn = self.input_rnn(self.external_input_seq_embed, d0)
        sampled_state = []
        for t in range(l):
            prior_z_t_mean, prior_z_t_logsigma = self.prior_gauss(
                torch.cat([z_t_minus_one, self.d_seq[t]], dim=-1)
            )
            z_t_dist = MultivariateNormal(prior_z_t_mean, logsigma2cov(prior_z_t_logsigma))
            z_t_minus_one = normal_differential_sample(z_t_dist)
            sampled_state.append(z_t_minus_one)
        self.sampled_state = torch.stack(sampled_state, dim=0)

        observations_dist = self.decode_observation(mode='dist')

        # 对观测分布采样并更新后验lstm隐状态
        if max_prob:
            observations_sample = observations_dist.loc
        else:
            observations_sample = normal_differential_sample(observations_dist)

        return observations_dist, observations_sample, {'dn': dn, 'zn': z_t_minus_one}

    def call_loss(self):

        D = self.D if self.training else 1
        l, batch_size, _ = self.observations_seq.shape
        memory_state = self.memory_state
        z_t_minus_one = torch.zeros((batch_size, self.state_size), device=self.external_input_seq.device) if memory_state is None else memory_state['zn']

        z_t_minus_one_seq = torch.cat([z_t_minus_one.unsqueeze(0), self.sampled_state[:-1]], dim=0)

        # prior_z_t_seq_mean, prior_z_t_seq_logsigma = self.prior_gauss(
        #     torch.cat([z_t_minus_one_seq, self.d_seq], dim=-1)
        # )
        #
        # kl_sum = multivariat_normal_kl_loss(
        #     self.state_mu,
        #     logsigma2cov(self.state_logsigma),
        #     prior_z_t_seq_mean,
        #     logsigma2cov(prior_z_t_seq_logsigma)
        # )

        kl_sum = 0

        predicted_state_sampled = z_t_minus_one_seq  # shape with (l,bs,state_size)
        for step in range(D):

            length = predicted_state_sampled.size()[0]
            prior_z_t_seq_mean, prior_z_t_seq_logsigma = self.prior_gauss(
                torch.cat([predicted_state_sampled, self.d_seq[-length:]], dim=-1)
            )

            kl_sum += multivariate_normal_kl_loss(
                self.state_mu[-length:],
                logsigma2cov(self.state_logsigma[-length:]),
                prior_z_t_seq_mean,
                logsigma2cov(prior_z_t_seq_logsigma)
            )
            predicted_state_sampled = normal_differential_sample(
                MultivariateNormal(
                    prior_z_t_seq_mean, logsigma2cov(prior_z_t_seq_logsigma)
                )
            )[:-1]

        kl_sum = kl_sum/D

        # kl_sum = 0
        # kl_list = []
        # for t in range(l):
        #     prior_z_t_mean, prior_z_t_logsigma = self.prior_gauss(
        #         torch.cat([z_t_minus_one, self.d_seq[t]], dim=-1)
        #     )
        #     kl = multivariat_normal_kl_loss(
        #         self.state_mu[t],
        #         logsigma2cov(self.state_logsigma[t]),
        #         prior_z_t_mean,
        #         logsigma2cov(prior_z_t_logsigma)
        #     )
        #     kl_sum += kl
        #     z_t_minus_one = self.sampled_state[t]
        #     kl_list.append(kl)
        #
        # print(kl_list)
        observations_normal_dist = self.decode_observation(mode='dist')

        generative_likelihood = torch.sum(observations_normal_dist.log_prob(self.observations_seq))

        return (kl_sum - generative_likelihood)/batch_size, kl_sum/batch_size, -generative_likelihood/batch_size

    def decode_observation(self, mode='sample'):
        """

        Args:
            state: with shape (len, batch_size, state_size)
            mode: dist or sample

        Returns:

        """
        mean, logsigma = self.decoder(
            torch.cat([self.sampled_state, self.d_seq], dim=-1)
        )
        observations_normal_dist = MultivariateNormal(
            mean, logsigma2cov(logsigma)
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return observations_normal_dist.sample()

