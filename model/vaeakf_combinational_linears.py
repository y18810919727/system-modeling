#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
from common import softplus, inverse_softplus, cov2logsigma, logsigma2cov, split_first_dim, merge_first_two_dims
from model.func import weighted_linear, normal_differential_sample, multivariate_normal_kl_loss
from model.dynamic.combinational_linears import CombinationalLinears
from model.common import DBlock, PreProcess
from model.common import DiagMultivariateNormal as MultivariateNormal

class VAEAKFCombinedLinear(nn.Module):
    def __init__(self, input_size, state_size, observations_size, net_type='lstm', k=16, num_layers=1, L=1, R=1,
                 num_linears=8,
                 D=30):

        """

        :param input_size: The size of external input
        :param state_size:
        :param observations_size:
        :param net_type: the type of VAE
        :param k: size of hidden state
        :param num_layers:
        :param L: The times for sampling auxiliary distribution.
        """
        super(VAEAKFCombinedLinear, self).__init__()
        self.k = k
        self.observations_size = observations_size
        self.state_size = state_size
        self.input_size = input_size
        self.L = L
        self.R = R
        self.D = D
        self.num_layers = num_layers

        assert D >= 1 and L >= 1 and R >= 1
        # region Define parameters for training
        self.process_x = PreProcess(observations_size, k)
        self.process_u = PreProcess(input_size, k)
        self.net_type = net_type
        self.rnn = nn.LSTM(k*2, k, num_layers=num_layers)
        self.posterior_hidden_state0 = torch.nn.Parameter(torch.zeros(k))
        self.gauss_decoder = DBlock(k, k*2, state_size)

        # In combinated linears, learnable linear matrix are weighted upon a softmax output.
        self.dynamic = CombinationalLinears(
            k,
            state_size,
            num_layers=num_layers,
            num_linears=num_linears,
            hidden_size=k
        )

        self.decoder = DBlock(state_size, k, observations_size)

        # endregion

        self.observations_seq_embed = None
        self.initial_prior_mu, self.initial_prior_logsigma = None, None
        self.sampled_state = None
        self.weight_initial_hidden_state = None

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        if memory_state is None:
            posterior_lstm_state = None
            weight_initial_hidden_state = None
        else:
            posterior_lstm_state = memory_state['posterior_lstm_state']
            weight_initial_hidden_state = memory_state['weight_hidden_state']

        external_input_seq_embed = self.process_u(external_input_seq)
        observations_seq_embed = self.process_x(observations_seq)

        l, batch_size, _ = external_input_seq.size()

        posterior_lstm_state =  self.generate_lstm_initial_state(
            self.posterior_hidden_state0, batch_size, self.num_layers) if posterior_lstm_state is None else posterior_lstm_state

        initial_prior_mu, initial_prior_logsigma = self.gauss_decoder(
            posterior_lstm_state[0][-1]
        )

        all_seq = torch.cat([external_input_seq_embed, observations_seq_embed], dim=2)
        z_seq, new_posterior_lstm_state = self.rnn(all_seq, posterior_lstm_state)
        state_mu, state_logsigma = self.gauss_decoder(z_seq)

        self.weight_initial_hidden_state = weight_initial_hidden_state
        # region 估计每个位置的weight_hidden_state
        sampled_state = normal_differential_sample(
            MultivariateNormal(state_mu, logsigma2cov(state_logsigma))
        )
        # weight_initial_hidden_state 是上一轮次forward时lstm的隐状态(hn, cn)

        weight_hidden_state = self.generate_lstm_initial_state(
            self.dynamic.weight_hidden_state0, batch_size, self.num_layers
        ) if weight_initial_hidden_state is None else weight_initial_hidden_state

        _, weight_hidden_state = self.dynamic.linears_weight_lstm(
            sampled_state, weight_hidden_state
        )
        new_weight_initial_hidden_state = weight_hidden_state
        # endregion

        outputs = {
            'initial_prior_mu': initial_prior_mu,
            'initial_prior_logsigma': initial_prior_logsigma,
            'state_mu': state_mu,
            'state_logsigma': state_logsigma,
            'sampled_state': sampled_state,
            'external_input_seq_embed': external_input_seq_embed
        }

        return outputs, {
            'posterior_lstm_state': new_posterior_lstm_state, 'weight_hidden_state': new_weight_initial_hidden_state
        }

    def forward_prediction(self, external_input_seq, max_prob=False, memory_state=None):

        if memory_state is None:
            posterior_lstm_state = None
            weight_initial_hidden_state = None
        else:
            posterior_lstm_state = memory_state['posterior_lstm_state']
            weight_initial_hidden_state = memory_state['weight_hidden_state']

        external_input_seq_embed = self.process_u(external_input_seq)
        l, batch_size, _ = external_input_seq.size()

        # 生成-1时刻的隐变量分布
        posterior_lstm_state = self.generate_lstm_initial_state(
            self.posterior_hidden_state0, batch_size, self.num_layers) if posterior_lstm_state is None else posterior_lstm_state

        weight_hidden_state = self.generate_lstm_initial_state(
            self.dynamic.weight_hidden_state0, batch_size, self.num_layers) if weight_initial_hidden_state is None else weight_initial_hidden_state

        state_sampled_list = []
        weight_map_list = []

        initial_prior_mu, initial_prior_logsigma = self.gauss_decoder(
            posterior_lstm_state[0][-1]
        )

        # 预测过程通过ancestral sampling方法采样l个时刻的隐状态

        # 此处采样-1时刻的隐状态
        state = normal_differential_sample(
            MultivariateNormal(initial_prior_mu, logsigma2cov(initial_prior_logsigma))
        )
        for i in range(l):

            # 根据weight_hidden_state计算i-1位置隐状态对应的linears weight, lstm为多层时，取h的最后一层状态
            next_state_dist, state, (weight_hidden_state, weight_map) = self.dynamic(
                state, external_input_seq_embed[i], (weight_hidden_state, ))
            state_sampled_list.append(state)
            weight_map_list.append(weight_map)

        sampled_state = torch.stack(state_sampled_list)
        weight_map = torch.stack(weight_map_list)

        # 利用隐状态计算观测数据分布
        observations_dist = self.decode_observation({
            'sampled_state': sampled_state
        }, mode='dist')

        # 对观测分布采样并更新后验lstm隐状态
        if max_prob:
            observations_sample = observations_dist.loc
        else:
            observations_sample = normal_differential_sample(observations_dist)
        _, posterior_lstm_state = self.rnn(
            torch.cat([external_input_seq_embed, self.process_x(observations_sample)], dim=-1), posterior_lstm_state
        )
        outputs = {
            'predicted_dist': observations_dist,
            'predicted_seq': observations_sample
        }
        return outputs, {
            'posterior_lstm_state': posterior_lstm_state,
            'weight_hidden_state': weight_hidden_state,
            'weight_map': weight_map
        }

    @staticmethod
    def generate_lstm_initial_state(state0, batch_size, num_layers):
        """

        Args:
            state0: 可学习的初始隐状态 hn
            state0:
            batch_size:

        Returns:
            扩展hn为多层，并产生全零cn
            如果posterior_lstm_state 为 None 根据模型内部参数重构posterior_lstm_state，并解码先验分布
            否则直接对posterior_lstm_state解码
        """
        initial_hidden_state = torch.cat([
            torch.zeros_like(state0).repeat(num_layers-1, batch_size, 1),
            state0.repeat(batch_size, 1).unsqueeze(dim=0)
        ], dim=0)
        lstm_state = (initial_hidden_state, torch.zeros_like(initial_hidden_state))

        return lstm_state

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        """

        Returns: 调用call_loss前需要调用forward_posterior，以获得输入数据及隐状态的后验分布。
        该算法采用 stochastic overshooting 方法来估计隐变量后验分布与先验分布的kl散度。具体过程如下：
        对于任意位置i,其后验分布为q(i)， 利用后验分布采样q(i-d)，其中d为[1,D)的随机值。利用祖先采样从分布p(i-1|i-d, a[i-d:i-1])采样。
        然后根据采样出来的h(i-1)计算i时刻隐变量的先验分布p(i|i-1)，并计算与q(i)的kl散度。

        """
        outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)

        # outputs ={
        #     'initial_prior_mu': initial_prior_mu,
        #     'initial_prior_logsigma': initial_prior_logsigma,
        #     'state_mu': state_mu,
        #     'state_logsigma': state_logsigma,
        #     'sampled_state': sampled_state
        # }
        initial_prior_mu = outputs['initial_prior_mu']
        initial_prior_logsigma = outputs['initial_prior_logsigma']
        state_mu = outputs['state_mu']
        state_logsigma = outputs['state_logsigma']
        weight_initial_hidden_state = memory_state['weight_hidden_state']
        external_input_seq_embed = outputs['external_input_seq_embed']

        l, bs, _ = observations_seq.shape
        if self.training:
            D = self.D
        else:
            D = 1

        # system dynamic: p(s(t+1) | s(t), a(t+1))
        # o(t)        o(t+1)
        #  ^             ^
        #  |             |
        #  |             |
        # s(t) -----> s(t+1)
        #                ^
        #                |
        #                |
        #             a(t+1)

        # region calculating kl divergence :

        # 把系统-1时刻的先验分布加到后验分布中
        q_mu = torch.cat([initial_prior_mu.unsqueeze(dim=0), state_mu], dim=0)
        q_cov = torch.cat([logsigma2cov(initial_prior_logsigma).unsqueeze(dim=0), logsigma2cov(state_logsigma)])

        # 先从后验分布中采样，长度为 l + 1
        sampled_state_l_plus_one = normal_differential_sample(
            MultivariateNormal(q_mu, q_cov)
        )

        # 取forward_posterior时存储的weight_initial_hidden_state构建lstm的初始隐状态(hn,cn)，如果为None，从动态模型dynamic内置参数构建
        weight_hidden_state = self.generate_lstm_initial_state(
            self.dynamic.weight_hidden_state0, bs, self.num_layers
        ) if weight_initial_hidden_state is None else weight_initial_hidden_state

        # region 计算各个位置的weight lstm 隐状态
        weight_h_c_list = [weight_hidden_state]
        for i in range(1, l+1):
            _, weight_hidden_state = self.dynamic.linears_weight_lstm(
                sampled_state_l_plus_one[i:i+1], weight_hidden_state
            )
            weight_h_c_list.append(weight_hidden_state)

        # weight_hidden_state_memory : A tuple (l+1, bs, num_layers,  h_size), (l+1, bs, num_layers, c_size)
        weight_hidden_state_memory = tuple([torch.stack(state).contiguous().transpose(2, 1) for state in zip(
            *weight_h_c_list
        )])

        # region 魔法开始! latent overshooting

        """
        定义名词:
        训练位置 : i, 指计算KL[q(i)||p(i|i-1)]的位置
        起始位置: i-t ,指over shooting 中做长序列预测的起始位置p(i-1|i-t)，其中t对于不同batch中的不同训练数据、不同序列位置i都是随机的
        单步预测位置: i-1
        """
        kl_list = []
        for _ in range(self.R):

            sum_kl = 0
            predicted_state_sampled = sampled_state_l_plus_one[:-1]  # shape with (l,bs,state_size)
            recursive_weight_hidden_state = tuple([weight_state[:-1] for weight_state in weight_hidden_state_memory])
            for step in range(D):
                length = predicted_state_sampled.size()[0]
                hn, cn = recursive_weight_hidden_state
                # (length, bs, num_layers, c_size) -> (num_layers, length*bs, c_size) ,
                # lstm输入隐状态要求layers在前，batch_size在后
                hn = merge_first_two_dims(hn).contiguous().transpose(1, 0)
                cn = merge_first_two_dims(cn).contiguous().transpose(1, 0)

                external_input_need = external_input_seq_embed[-length:]
                next_state_dist, next_state_sample, ((new_hn, new_cn), weight_map) = self.dynamic(
                    merge_first_two_dims(predicted_state_sampled),
                    merge_first_two_dims(external_input_need), ((hn, cn),)
                )
                sum_kl += multivariate_normal_kl_loss(state_mu[-length:],
                                                     logsigma2cov(state_logsigma[-length:]),
                                                     split_first_dim(next_state_dist.loc, (length, bs)),
                                                     split_first_dim(next_state_dist.covariance_matrix, (length, bs))
                                                     )
                predicted_state_sampled = split_first_dim(next_state_sample, (length, bs))[:-1]
                recursive_weight_hidden_state = tuple([
                    split_first_dim(s.transpose(1, 0), (length, bs))[:-1] for s in [new_hn, new_cn]
                ])

            kl_list.append(sum_kl/D)

        kl = sum(kl_list) / self.R
        # endregion

        # region calculation generative probability p(x|z)
        generative_likelihood = self.estimate_generative_logprobability(state_mu, state_logsigma, observations_seq)
        # endregion

        # maximun -kl + generative_likelihood
        return {
            'loss': (kl - generative_likelihood)/bs,
            'kl_loss': kl/bs,
            'likelihood_loss': -generative_likelihood/bs
        }

    def estimate_generative_logprobability(self, state_mu, state_logsigma, observations_seq):

        generative_likelihood = []
        for _ in range(self.L):
            state = self.sample_state(state_mu, state_logsigma, max_prob=False)
            observations_mu, observations_cov = self.decoder(state)
            # observations_mu = torch.nn.functional.linear(state, self.estimate_H) + self.estimate_bias
            # observations_cov = logsigma2cov(self.estimate_logdelta)
            observations_normal_dist = MultivariateNormal(
                observations_mu, logsigma2cov(observations_cov)
            )
            generative_likelihood.append(torch.sum(observations_normal_dist.log_prob(observations_seq)))

        generative_likelihood = sum(generative_likelihood)/self.L
        return generative_likelihood

    def decode_observation(self, outputs, mode='sample'):
        """

        Args:
            state: with shape (len, batch_size, state_size)
            mode: dist or sample

        Returns:

        """

        observations_mu, observations_cov = self.decoder(outputs['sampled_state'])
        observations_normal_dist = MultivariateNormal(
            observations_mu, logsigma2cov(observations_cov)
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return observations_normal_dist.sample()

    def sample_state(self, state_mu, state_logsigma, max_prob=True):
        """
        :param max_prob: If True, return the sequence which has the biggest probability. Otherwise, a randomly sampled
        sequence will be returned.
        :return:(l, bs, state_size)
        """
        if state_mu is None:
            raise AssertionError('You have to call forward method before sampling state')
        if max_prob:
            return state_mu
        noise = torch.randn_like(state_mu)
        state = noise*softplus(state_logsigma) + state_mu
        return state

    # def sigma_interval(self, e):
    #     return self.state_mu - e * softplus(self.state_logsigma), \
    #            self.state_mu + e * softplus(self.state_logsigma)

