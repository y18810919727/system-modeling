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
                 num_linears=8,random_overshooting=True,
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
        self.random_overshooting = random_overshooting
        self.num_layers = num_layers

        assert D >= 1 and L >= 1 and R >= 1
        # region Define parameters for training
        self.process_x = PreProcess(input_size + observations_size, k)
        self.net_type = net_type
        self.rnn = nn.LSTM(k, k, num_layers=num_layers)
        self.posterior_hidden_state0 = torch.nn.Parameter(torch.zeros(k))
        self.gauss_decoder = DBlock(k, k*2, state_size)

        # In combinated linears, learnable linear matrix are weighted upon a softmax output.
        self.dynamic = CombinationalLinears(
            input_size,
            state_size,
            num_layers=num_layers,
            num_linears=num_linears,
            hidden_size=k
        )

        self.estimate_logdelta = torch.nn.parameter.Parameter(torch.randn(observations_size))  # noise in observation
        self.estimate_H = torch.nn.parameter.Parameter(torch.zeros((observations_size, state_size)))
        self.estimate_bias = torch.nn.parameter.Parameter(torch.randn(observations_size)) # bias in observation

        # endregion

        self.state_mu = None
        self.state_logsigma = None
        self.external_input_seq = None
        self.observations_seq = None
        self.initial_prior_mu, self.initial_prior_logsigma = None, None
        self.sampled_state = None
        self.weight_initial_hidden_state = None

    def forward_posterior(self, external_input_seq, observations_seq, posterior_lstm_state=None, weight_initial_hidden_state=None):

        self.external_input_seq = external_input_seq
        self.observations_seq = observations_seq

        l, batch_size, _ = external_input_seq.size()

        posterior_lstm_state =  self.generate_lstm_initial_state(
            self.posterior_hidden_state0, batch_size, self.num_layers) if posterior_lstm_state is None else posterior_lstm_state

        self.initial_prior_mu, self.initial_prior_logsigma = self.gauss_decoder(
            posterior_lstm_state[0][-1]
        )

        all_seq = torch.cat([external_input_seq, observations_seq], dim=2)
        all_seq = self.process_x(all_seq)
        z_seq, new_posterior_lstm_state = self.rnn(all_seq, posterior_lstm_state)
        self.state_mu, self.state_logsigma = self.gauss_decoder(z_seq)

        self.weight_initial_hidden_state = weight_initial_hidden_state
        # region 估计每个位置的weight_hidden_state
        sampled_state = normal_differential_sample(
            MultivariateNormal(self.state_mu, logsigma2cov(self.state_logsigma))
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

        return self.state_mu, self.state_logsigma, new_posterior_lstm_state, new_weight_initial_hidden_state

    def forward_prediction(self, external_input_seq, posterior_lstm_state=None, weight_initial_hidden_state=None,
                           max_prob=False, return_weight_map=False):

        l, batch_size, _ = external_input_seq.size()

        # 生成-1时刻的隐变量分布
        posterior_lstm_state = self.generate_lstm_initial_state(
            self.posterior_hidden_state0, batch_size, self.num_layers) if posterior_lstm_state is None else posterior_lstm_state

        weight_hidden_state = self.generate_lstm_initial_state(
            self.dynamic.weight_hidden_state0, batch_size, self.num_layers) if weight_initial_hidden_state is None else weight_initial_hidden_state

        state_sampled_list = []
        weight_map_list = []

        self.initial_prior_mu, self.initial_prior_logsigma = self.gauss_decoder(
            posterior_lstm_state[0][-1]
        )

        # 预测过程通过ancestral sampling方法采样l个时刻的隐状态

        # 此处采样-1时刻的隐状态
        state = normal_differential_sample(
            torch.distributions.MultivariateNormal(self.initial_prior_mu, logsigma2cov(self.initial_prior_logsigma))
        )

        for i in range(l):

            # 根据weight_hidden_state计算i-1位置隐状态对应的linears weight, lstm为多层时，取h的最后一层状态
            next_state_dist, state, (weight_hidden_state, weight_map) = self.dynamic(
                state, external_input_seq[i], (weight_hidden_state, ))
            state_sampled_list.append(state)
            weight_map_list.append(weight_map)


        sampled_state = torch.stack(state_sampled_list)
        weight_map = torch.stack(weight_map_list)

        # 利用隐状态计算观测数据分布
        observations_dist = self.decode(state=sampled_state, mode='dist')

        # 对观测分布采样并更新后验lstm隐状态
        if max_prob:
            observations_sample = observations_dist.loc
        else:
            observations_sample = normal_differential_sample(observations_dist)
        _, posterior_lstm_state = self.rnn(
            self.process_x(torch.cat([external_input_seq, observations_sample], dim=-1)), posterior_lstm_state
        )

        if return_weight_map:
            return observations_dist, observations_sample, posterior_lstm_state, weight_hidden_state, weight_map
        else:
            return observations_dist, observations_sample, posterior_lstm_state, weight_hidden_state


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


    def call_loss(self):
        """

        Returns: 调用call_loss前需要调用forward_posterior，以获得输入数据及隐状态的后验分布。
        该算法采用 stochastic overshooting 方法来估计隐变量后验分布与先验分布的kl散度。具体过程如下：
        对于任意位置i,其后验分布为q(i)， 利用后验分布采样q(i-d)，其中d为[1,D)的随机值。利用祖先采样从分布p(i-1|i-d, a[i-d:i-1])采样。
        然后根据采样出来的h(i-1)计算i时刻隐变量的先验分布p(i|i-1)，并计算与q(i)的kl散度。

        """

        l, bs, _  = self.observations_seq.shape

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
        q_mu = torch.cat([self.initial_prior_mu.unsqueeze(dim=0), self.state_mu], dim=0)
        q_cov = torch.cat([logsigma2cov(self.initial_prior_logsigma).unsqueeze(dim=0), logsigma2cov(self.state_logsigma)])

        # 先从后验分布中采样，长度为 l + 1
        sampled_state = normal_differential_sample(
            torch.distributions.MultivariateNormal(q_mu, q_cov)
        )

        # 取forward_posterior时存储的weight_initial_hidden_state构建lstm的初始隐状态(hn,cn)，如果为None，从动态模型dynamic内置参数构建
        weight_hidden_state = self.generate_lstm_initial_state(
            self.dynamic.weight_hidden_state0, bs, self.num_layers
        ) if self.weight_initial_hidden_state is None else self.weight_initial_hidden_state

        # region 计算各个位置的weight lstm 隐状态
        weight_h_c_list = [weight_hidden_state]
        for i in range(1, l+1):
            _, weight_hidden_state = self.dynamic.linears_weight_lstm(
                sampled_state[i:i+1], weight_hidden_state
            )
            weight_h_c_list.append(weight_hidden_state)

        # weight_hidden_state_memory : A tuple (l, bs, num_layers,  h_size), (l,bs, num_layers, c_size)
        weight_hidden_state_memory = tuple([torch.stack(state).contiguous().transpose(2, 1) for state in zip(
            *weight_h_c_list
        )])

        # region 魔法开始! random overshooting

        """
        定义名词:
        训练位置 : i, 指计算KL[q(i)||p(i|i-1)]的位置
        起始位置: i-t ,指over shooting 中做长序列预测的起始位置p(i-1|i-t)，其中t对于不同batch中的不同训练数据、不同序列位置i都是随机的
        单步预测位置: i-1
        """
        def kl_sample():


            for overshooting_length in (range(self.D, self.D+1) if self.random_overshooting else range(1, self.D+1)):
                # 不对完整的序列计算kl散度，因为需要往前最多look back D步,所以只考虑序列最后 l + 1 - D个位置，
                # trained_positions存储被训练位置的下标，以后要用下标来做gather操作

                trained_positions = torch.arange(overshooting_length, q_mu.size()[0], device=sampled_state.device).reshape(-1, 1).repeat(1, bs)
                trained_length = trained_positions.size()[0]
                if self.random_overshooting:
                    # 对每个待训练的位置，随机look back一定距离，随机范围是[1,D]
                    sampled_d = torch.randint(1, self.D+1, trained_positions.size(), device=sampled_state.device)
                else:
                    sampled_d = torch.randint(overshooting_length, overshooting_length + 1,
                                              trained_positions.size(), device=sampled_state.device)

                # 根据随机出来的距离，计算后验分布做采样的位置
                positions_minus_d = trained_positions - sampled_d

                def seq_gather(input, index):
                    """
                    根据index存储的下标，取input序列指定位置的张量
                    Args:
                        input: shape(length, bs, a, b, c, ...)
                        index: (l, bs)

                    Returns: tensor with shape (l, bs, a, b, c, ...)

                    """
                    l, bs = index.size()[0], index.size()[1]

                    assert (index >= 0).all() and (index < input.size()[0]).all()

                    return torch.gather(
                        input, dim=0, index=index.reshape(
                            l, bs, *([1]*(len(input.shape)-2))
                        ).repeat(1, 1, *input.size()[2:])
                    )

                assert (positions_minus_d >= 0).all()
                # 从sampled_state中取出起始位置对应的采样的状态
                predicted_state_sampled = seq_gather(sampled_state, positions_minus_d)
                # 取出起始位置对应的weight lstm 隐状态：用weight_hidden_state_memory内的状态作为初始状态，之后的状态边采样，边用lstm计算
                weight_hidden_state = tuple([seq_gather(weight_hidden_state_memory[_], positions_minus_d) for _ in [0, 1]])
                current_sampled_positions = positions_minus_d

                # 从起始位置开始，不停地做单步预测，直到预测到单步预测位置为止
                while (current_sampled_positions < trained_positions - 1).any():

                    # 找到那些还没到单步预测位置的positions
                    updated_positions = (current_sampled_positions < trained_positions - 1)
                    # 取出这些位置的采样状态
                    changed_state = predicted_state_sampled[updated_positions]
                    # 取出这些位置的weight lstm隐状态
                    changed_weight_hidden_state = tuple([weight_hidden_state[_][updated_positions] for _ in range(2)])
                    hn, cn = changed_weight_hidden_state
                    hn = hn.contiguous().transpose(1, 0)
                    cn = cn.contiguous().transpose(1, 0)

                    """
                    对于外部输入项下标为什么是current_sampled_positions的解释：
                    因为考虑了-1位置的先验分布，因此sampled_state，即完整后验隐状态采样的长度为l+1，对应下标位[0,l]，而外部输入序列
                    external_input_seq的长度为l,对应下标[0,l-1]，对应系统变换为 func(state[0] , input[0]) -> state[1]，因此，
                    current_sampled_positions存储的是当前正在向前滚动的隐状态下标，与其产生作用的外部输入序列下标应同为current_sampled_positions
                    """
                    external_input_need = seq_gather(self.external_input_seq, current_sampled_positions)[updated_positions]

                    # 利用动态系统模拟状态变化，返回下一时刻的分布及采样
                    next_state_dist, next_state_sample, ((new_hn, new_cn), weight_map) = self.dynamic(
                        changed_state, external_input_need, ((hn, cn),)
                    )
                    # 更新 状态、lstm隐状态
                    weight_hidden_state[0][updated_positions] = hn.contiguous().transpose(1, 0)
                    weight_hidden_state[1][updated_positions] = cn.contiguous().transpose(1, 0)
                    predicted_state_sampled[updated_positions] = next_state_sample

                    # 没到单步预测位置的位置增1
                    current_sampled_positions[updated_positions] += 1
                assert (current_sampled_positions + 1 == trained_positions).all()

                # 以单步预测位置作为起点，再做一次单步预测

                # dynamic 的输入tensor要求为(bs, xxx)，需要把张量前两维合并
                weight_hidden_state = tuple([merge_first_two_dims(weight_hidden_state[_]) for _ in range(2)])

                # 取其中单步预测分布, 其他返回值没用
                hn, cn = weight_hidden_state
                hn = hn.contiguous().transpose(1, 0)
                cn = cn.contiguous().transpose(1, 0)
                one_step_predicted_dist, _, _ = self.dynamic(merge_first_two_dims(predicted_state_sampled),
                                                             merge_first_two_dims(seq_gather(self.external_input_seq, current_sampled_positions)),
                                                             ((hn, cn),))
                # 计算训练位置i,后验分布q(i)与先验分布p(i|i-1)的kl散度
                kl = multivariate_normal_kl_loss(self.state_mu[-trained_length:],
                                                     logsigma2cov(self.state_logsigma[-trained_length:]),
                                                     split_first_dim(one_step_predicted_dist.loc, (trained_length, bs)),
                                                     split_first_dim(one_step_predicted_dist.covariance_matrix, (trained_length, bs))
                                                     )
            return kl
        kl = sum([kl_sample() for _ in range(self.R)])/self.R
        # endregion

        # region calculation generative probability p(x|z)
        generative_likelihood = self.estimate_generative_logprobability()
        # endregion

        # maximun -kl + generative_likelihood
        return (kl - generative_likelihood)/bs, kl/bs, -generative_likelihood/bs

    def estimate_generative_logprobability(self):

        def estimate_generative_logprobability_from_sample():
            state = self.sample_state(max_prob=False)
            observations_mu = torch.nn.functional.linear(state, self.estimate_H) + self.estimate_bias
            observations_cov = logsigma2cov(self.estimate_logdelta)
            observations_normal_dist = torch.distributions.MultivariateNormal(observations_mu, observations_cov)
            return torch.sum(observations_normal_dist.log_prob(self.observations_seq))

        generative_likelihood = sum([estimate_generative_logprobability_from_sample() for _ in range(self.L)])/self.L
        return generative_likelihood

    def decode(self, state, mode='sample'):
        """

        Args:
            state: with shape (len, batch_size, state_size)
            mode: dist or sample

        Returns:

        """
        observations_mu = torch.nn.functional.linear(state, self.estimate_H) + self.estimate_bias
        observations_cov = logsigma2cov(self.estimate_logdelta)
        observations_normal_dist = torch.distributions.MultivariateNormal(observations_mu, observations_cov)
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return observations_normal_dist.sample()

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

