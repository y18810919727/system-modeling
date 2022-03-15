#!/usr/bin/python
# -*- coding:utf8 -*-

import torch
from torch import nn
from model.common import DBlock, PreProcess
from common import logsigma2cov, split_first_dim, merge_first_two_dims
from model.common import DiagMultivariateNormal as MultivariateNormal
from model.func import normal_differential_sample, multivariate_normal_kl_loss
from model.ct_model import ODE_RNN
from model.vrnn import VRNN


class ODEVRNN(nn.Module):

    def __init__(self, input_size, state_size, observations_size, rnn_type='ode_rnn',
                 ode_solver='dopri5', k=16, D=5, ode_hidden_dim=50, rtol=1e-3, atol=1e-4):

        super(ODEVRNN, self).__init__()

        input_size = input_size - 1  # The last dimension of input variable represents dt
        self.k = k
        self.D = D
        self.observations_size = observations_size
        self.state_size = state_size
        self.input_size = input_size
        if rnn_type == 'ode_rnn':
            self.ode_rnn = ODE_RNN(3*k, k, ode_solver=ode_solver, ode_hidden_dim=ode_hidden_dim, rtol=rtol, atol=atol)
        else:
            raise NotImplementedError('The posterior ct model with type %d is not implemented!' % rnn_type)

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
        if memory_state is None:
            rnn_hidden_state = self.ode_rnn.init_hidden(batch_size, external_input_seq.device)
            h = rnn_hidden_state[0, ..., :self.ode_rnn.latent_dim]
        else:
            h, rnn_hidden_state = memory_state['hn'], memory_state['rnn_hidden']

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
                [observations_seq_embed[t], external_input_seq_embed[t], self.process_z(z_t), dt[t]], dim=-1
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

        if memory_state is None:
            rnn_hidden_state = self.ode_rnn.init_hidden(batch_size, external_input_seq.device)
            h = rnn_hidden_state[0, ..., :self.ode_rnn.latent_dim]
        else:
            h, rnn_hidden_state = memory_state['hn'], memory_state['rnn_hidden']

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
                [self.process_x(x_t), external_input_seq_embed[t], z_t_embed, dt[t]], dim=-1
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

        # external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]
        dt = external_input_seq[..., -1:]
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

            # TODO: 实现序列并行
            # region ct 模式下，序列不同位置的dt不同，序列并行计算较为复杂
            # output, rnn_hidden_state = self.rnn(
            #     merge_first_two_dims(
            #         torch.cat(
            #             [
            #                 self.process_x(x_t_seq),
            #                 external_input_seq_embed[-length:],
            #                 z_t_seq_embed,
            #                 dt[-length:]
            #             ], dim=-1
            #         )
            #     ).unsqueeze(dim=0),
            #     merge_first_two_dims(rnn_hidden_state_seq).contiguous().transpose(1, 0)
            # )
            # rnn_hidden_state_seq = split_first_dim(rnn_hidden_state.contiguous().transpose(1, 0), (length, batch_size))[:-1]
            # predicted_h = split_first_dim(output.squeeze(dim=0), (length, batch_size))[:-1]
            # endregion

            # TODO: 未来需要删掉
            # region ct 模式下，序列不同位置的dt不同，序列并行计算较为复杂
            output_list, rnn_hidden_state_list = [], []
            for i in range(length):
                output, rnn_hidden_state = self.ode_rnn(
                    torch.cat([
                        self.process_x(x_t_seq[i]),
                        external_input_seq_embed[-length:][i],
                        z_t_seq_embed[i],
                        dt[-length:][i]
                    ], dim=-1).unsqueeze(dim=0), rnn_hidden_state_seq[i].transpose(1, 0)
                )
                # Each tensor in list: (bs, latent_dim)
                output_list.append(output)
                # Each tensor in list: (bs, 1, 2*latent_dim)
                rnn_hidden_state_list.append(rnn_hidden_state.transpose(1, 0))
            predicted_h = torch.cat(output_list, dim=0)[:-1]
            rnn_hidden_state_seq = torch.stack(rnn_hidden_state_list)[:-1]
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
