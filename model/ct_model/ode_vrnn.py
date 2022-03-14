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
from model.common import DiagMultivariateNormal as MultivariateNormal
from model.func import normal_differential_sample, multivariate_normal_kl_loss, zeros_like_with_shape
from ct_model.ode_rnn import ODE_RNN
from model.vrnn import VRNN


class CTVRNN(VRNN):

    def __init__(self, input_size, state_size, observations_size, net_type='ode-rnn',
                 ode_solver='euler', k=16, num_layers=1, D=5):

        super(CTVRNN, self).__init__()

        self.k = k
        self.D = D
        self.observations_size = observations_size
        self.state_size = state_size
        self.input_size = input_size - 1  # The last dimension of input variable represents dt
        self.num_layers = num_layers
        if net_type == 'ode-rnn':
            self.ode_rnn = ODE_RNN(3*k, k, ode_solver=ode_solver)
        else:
            raise NotImplementedError('The posterior ct model with type %d is not implemented!' % net_type)

        self.process_u = PreProcess(input_size, k)
        self.process_x = PreProcess(observations_size, k)
        self.process_z = PreProcess(state_size, k)

        self.posterior_gauss = DBlock(3*k, 3*k, state_size)
        self.prior_gauss = DBlock(k+k, 3*k, state_size)
        self.decoder = DBlock(2*k, 3*k, observations_size)

        # self.state_mu = None
        # self.state_logsigma = None
        # self.external_input_seq = None
        # self.observations_seq = None
        # self.initial_prior_mu, self.initial_prior_logsigma = None, None
        # self.sampled_state = None
        self.weight_initial_hidden_state = None
        self.h_seq = None
        self.external_input_seq_embed = None
        self.observations_seq_embed = None
        self.memory_state = None
        self.rnn_hidden_state_seq = None

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        """
        训练时：估计隐变量后验分布，并采样，用于后续计算模型loss
        测试时: 为后续执行forward_prediction计算memory_state(h, rnn_hidden)
        Args:
            external_input_seq: 系统输入序列(进出料浓度、流量) (len, batch_size, input_size)
            observations_seq: 观测序列(泥层压强) (len, batch_size, observations_size)
            memory_state: 模型支持长序列预测，之前forward过程产生的记忆信息压缩在memory_state中

        Returns:

        """

        external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]

        external_input_seq_embed = self.process_u(external_input_seq)
        observations_seq_embed = self.process_x(observations_seq)

        l, batch_size, _ = external_input_seq.size()

        # 构建ode_rnn网络的初始隐状态和h，如果memory_state里有，从memory_state里拿
        # h, rnn_hidden_state = (
        #     zeros_like_with_shape(external_input_seq, (batch_size, self.k)),
        #     zeros_like_with_shape(external_input_seq, (self.num_layers, batch_size, self.k))
        # ) if memory_state is None else (memory_state['hn'], memory_state['rnn_hidden'])
        rnn_hidden_state = self.ode_rnn.init_hidden(batch_size, external_input_seq.device)
        h = rnn_hidden_state[0, ..., :self.ode_rnn.latent_dim]

        state_mu = []
        state_logsigma = []
        sampled_state = []
        h_seq = [h]
        rnn_hidden_state_seq = [rnn_hidden_state.transpose(1, 0)]
        for t in range(l):

            # 估计每一时刻t下，z的后验分布
            z_t_mean, z_t_logsigma = self.posterior_gauss(
                torch.cat([observations_seq_embed[t], external_input_seq_embed[t], h], dim=-1)
            )
            # 从分布做采样得到z_t
            z_t = normal_differential_sample(
                MultivariateNormal(z_t_mean, logsigma2cov(z_t_logsigma))
            )

            #更新rnn隐状态和h
            output, rnn_hidden_state = self.ode_rnn(torch.cat(
                [observations_seq_embed[t], external_input_seq_embed[t], self.process_z(z_t), dt], dim=-1
            ).unsqueeze(dim=0), rnn_hidden_state)
            h = output[0]

            # 记录第t时刻的z的分布，z的采样，h，rnn隐状态
            state_mu.append(z_t_mean)
            state_logsigma.append(z_t_logsigma)
            sampled_state.append(z_t)
            h_seq.append(h)
            rnn_hidden_state_seq.append(rnn_hidden_state.contiguous().transpose(1, 0))

        # 将list stack起来，构成torch的tensor
        state_mu = torch.stack(state_mu, dim=0)
        state_logsigma = torch.stack(state_logsigma, dim=0)
        sampled_state = torch.stack(sampled_state, dim=0)
        h_seq = torch.stack(h_seq, dim=0)
        rnn_hidden_state_seq = torch.stack(rnn_hidden_state_seq, dim=0)

        outputs = {
            'state_mu': state_mu,
            'state_logsigma': state_logsigma,
            'sampled_state': sampled_state,
            'h_seq': h_seq,
            'external_input_seq_embed': external_input_seq_embed,
            'rnn_hidden_state_seq': rnn_hidden_state_seq
        }
        return outputs, {'hn': h, 'rnn_hidden': rnn_hidden_state}

    def forward_prediction(self, external_input_seq, max_prob=False, memory_state=None):
        """
        给定模型记忆memory_state和系统外部输入序列(len,batch_size, input_size)估计系统未来输出的分布，
        并更新memory_state

        Args:
            external_input_seq:
            max_prob: 从预测的系统输出分布中进行采样的方法：max_prob为True，按最大概率采样；max_prob为False，随机采样;
            memory_state:

        Returns:

        """

        external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]

        l, batch_size, _ = external_input_seq.size()

        rnn_hidden_state = self.ode_rnn.init_hidden(batch_size, external_input_seq.device)
        h = rnn_hidden_state[0, ..., :self.ode_rnn.latent_dim]

        external_input_seq_embed = self.process_u(external_input_seq)

        sampled_state = []
        h_seq = [h]
        rnn_hidden_state_seq = [rnn_hidden_state.transpose(1, 0)]

        for t in range(l):

            z_t_mean, z_t_logsigma = self.prior_gauss(
                torch.cat([external_input_seq_embed[t], h], dim=-1)
            )
            z_t = normal_differential_sample(
                MultivariateNormal(z_t_mean, logsigma2cov(z_t_logsigma))
            )
            z_t_embed = self.process_z(z_t)
            x_t_mean, x_t_logsigma = self.decoder(
                torch.cat([z_t_embed, h], dim=-1)
            )
            x_t = normal_differential_sample(
                MultivariateNormal(x_t_mean, logsigma2cov(x_t_logsigma))
            )
            output, rnn_hidden_state = self.ode_rnn(torch.cat(
                [self.process_x(x_t), external_input_seq_embed[t], z_t_embed, dt], dim=-1
            ).unsqueeze(dim=0), rnn_hidden_state)
            h = output[0]

            sampled_state.append(z_t)
            h_seq.append(h)
            rnn_hidden_state_seq.append(rnn_hidden_state.contiguous().transpose(1, 0))

        sampled_state = torch.stack(sampled_state, dim=0)
        h_seq = torch.stack(h_seq, dim=0)  # with shape (l+1, bs, k)
        rnn_hidden_state_seq = torch.stack(rnn_hidden_state_seq, dim=0)

        observations_dist = self.decode_observation(
            {'sampled_state': sampled_state,
             'h_seq': h_seq},
            mode='dist'
        )

        # 对观测分布采样并更新后验lstm隐状态
        if max_prob:
            observations_sample = observations_dist.loc
        else:
            observations_sample = normal_differential_sample(observations_dist)

        outputs = {
            'sampled_state': sampled_state,
            'h_seq': h_seq,
            'rnn_hidden_state_seq': rnn_hidden_state_seq,
            'predicted_dist': observations_dist,
            'predicted_seq': observations_sample
        }
        return outputs, {'hn': h, 'rnn_hidden': rnn_hidden_state}

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        """
        此方法仅运行在调用完forward_posterior之后
        Returns:
        """

        external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]
        outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)

        h_seq = outputs['h_seq']
        rnn_hidden_state_seq = outputs['rnn_hidden_state_seq']
        state_mu = outputs['state_mu']
        state_logsigma = outputs['state_logsigma']
        external_input_seq_embed = outputs['external_input_seq_embed']

        l, batch_size, _ = observations_seq.shape

        D = self.D if self.training else 1

        kl_sum = 0

        predicted_h = h_seq[:-1]  # 删掉h_seq中的最后一个
        rnn_hidden_state_seq = rnn_hidden_state_seq[:-1]  # (length, bs, num_layers, k)

        # 此处为latent overshooting，d为over_shooting的距离，这部分代码思维强度有点深，需要多看几遍。
        for d in range(D):
            length = predicted_h.size()[0]

            # 预测的隐变量先验分布
            prior_z_t_seq_mean, prior_z_t_seq_logsigma = self.prior_gauss(
                torch.cat([external_input_seq_embed[-length:], predicted_h], dim=-1)
            )
            # 计算loss中的kl散度项
            kl_sum += multivariate_normal_kl_loss(
                state_mu[-length:],
                logsigma2cov(state_logsigma[-length:]),
                prior_z_t_seq_mean,
                logsigma2cov(prior_z_t_seq_logsigma)
            )

            # 利用reparameterization trick 采样z
            z_t_seq = normal_differential_sample(
                MultivariateNormal(prior_z_t_seq_mean, logsigma2cov(prior_z_t_seq_logsigma))
            )
            z_t_seq_embed = self.process_z(z_t_seq)
            x_t_seq_mean, x_t_seq_logsigma = self.decoder(
                torch.cat([z_t_seq_embed, predicted_h], dim=-1)
            )
            x_t_seq = normal_differential_sample(
                MultivariateNormal(x_t_seq_mean, logsigma2cov(x_t_seq_logsigma))
            )

            # region 这几行需要深入理解，其目的是将predicted_h往前推一格，并删除掉predicted_h的最后一项
            # rnn_hidden_state 's shape : (num_layers, length*batch_size, k)
            output, rnn_hidden_state = self.rnn(
                merge_first_two_dims(
                    torch.cat(
                        [
                            self.process_x(x_t_seq),
                            external_input_seq_embed[-length:],
                            z_t_seq_embed,
                            dt
                        ], dim=-1
                    )
                ).unsqueeze(dim=0),
                merge_first_two_dims(rnn_hidden_state_seq).contiguous().transpose(1, 0)
            )
            rnn_hidden_state_seq = split_first_dim(rnn_hidden_state.contiguous().transpose(1, 0), (length, batch_size))[:-1]
            predicted_h = split_first_dim(output.squeeze(dim=0), (length, batch_size))[:-1]
            # endregion

        kl_sum = kl_sum/D

        # prior_z_t_seq_mean, prior_z_t_seq_logsigma = self.prior_gauss(
        #     torch.cat([self.external_input_seq_embed, self.h_seq[:-1]], dim=-1)
        # )
        #
        # kl_sum = multivariate_normal_kl_loss(
        #     self.state_mu,
        #     logsigma2cov(self.state_logsigma),
        #     prior_z_t_seq_mean,
        #     logsigma2cov(prior_z_t_seq_logsigma)
        # )
        # 对h解码得到observation的generative分布
        observations_normal_dist = self.decode_observation(outputs, mode='dist')
        # 计算loss的第二部分：观测数据的重构loss
        generative_likelihood = torch.sum(observations_normal_dist.log_prob(observations_seq))

        return {
            'loss': (kl_sum - generative_likelihood)/batch_size,
            'kl_loss': kl_sum/batch_size,
            'likelihood_loss': -generative_likelihood/batch_size
        }

    def decode_observation(self, outputs, mode='sample'):
        """

        Args:
            state: with shape (len, batch_size, state_size)
            mode: dist or sample

        Returns:

        """
        mean, logsigma = self.decoder(
            torch.cat([
                self.process_z(outputs['sampled_state']), outputs['h_seq'][:-1]
            ], dim=-1)
        )
        observations_normal_dist = MultivariateNormal(
            mean, logsigma2cov(logsigma)
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return observations_normal_dist.sample()
