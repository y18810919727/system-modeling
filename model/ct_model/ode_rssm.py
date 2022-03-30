#!/usr/bin/python
# -*- coding:utf8 -*-

import torch
from torch import nn
from model.common import DBlock, PreProcess
from common import logsigma2cov, split_first_dim, merge_first_two_dims
from model.common import DiagMultivariateNormal as MultivariateNormal, MLP
from model.func import normal_differential_sample, multivariate_normal_kl_loss
from model.ct_model import DiffeqSolver,ODEFunc
from torch.nn import GRUCell
from model.vrnn import VRNN


class ODERSSM(nn.Module):

    def __init__(self, input_size, state_size, observations_size,
                 ode_solver='dopri5', k=16, D=5, ode_hidden_dim=50, ode_num_layers=1, rtol=1e-3, atol=1e-4,
                 ode_type='normal'):

        super(ODERSSM, self).__init__()

        input_size = input_size - 1  # The last dimension of input variable represents dt
        self.k = k
        self.D = D
        self.observations_size = observations_size
        self.state_size = state_size
        self.input_size = input_size
        self.gru_cell = GRUCell(2*k, 2*k)
        ode_func = ODEFunc(
            input_dim=k,
            ode_hidden_dim=ode_hidden_dim,
            ode_num_layers=ode_num_layers,
            ode_type=ode_type,
        )
        self.diffeq_solver = DiffeqSolver(k, ode_func, ode_solver,
                                          odeint_rtol=rtol, odeint_atol=atol)

        self.process_u = PreProcess(input_size, k)
        self.process_x = PreProcess(observations_size, k)
        self.process_z = PreProcess(state_size, k)

        self.posterior_gauss = DBlock(2*k, 3*k, state_size)
        self.prior_gauss = DBlock(k, 3*k, state_size)
        self.decoder = DBlock(2*k, 3*k, observations_size)

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
        device = external_input_seq.device
        l, batch_size, _ = external_input_seq.size()

        if memory_state is None:
            h, rnn_hidden_state = self.init_h(batch_size, device)
        else:
            h, rnn_hidden_state = memory_state['hn'], memory_state['rnn_hidden']

        state_mu = []
        state_logsigma = []
        sampled_state = []
        h_seq = [h]
        rnn_hidden_state_seq = [rnn_hidden_state]
        for i in range(l):

            # 估计每一时刻t下，z的后验分布
            z_t_mean, z_t_logsigma = self.posterior_gauss(
                torch.cat([observations_seq_embed[i], h], dim=-1)
            )
            # 从分布做采样得到z_t
            z_t = normal_differential_sample(
                MultivariateNormal(z_t_mean, logsigma2cov(z_t_logsigma))
            )

            h, rnn_hidden_state = self.GRU_update(torch.cat([
                external_input_seq_embed[i],
                self.process_z(z_t),
            ], dim=-1), rnn_hidden_state)

            # dt[i]代表 t[i+1] - t[i]，预测t[i+1]时刻的h
            h = self.diffeq_solver(h, torch.stack([torch.zeros_like(dt[i, :, 0]), dt[i, :, 0]]))[-1]
            rnn_hidden_state = self.update_rnn_hidden_state(h, rnn_hidden_state)

            h_seq.append(h)
            rnn_hidden_state_seq.append(rnn_hidden_state)

            # 记录第t时刻的z的分布，z的采样，h，rnn隐状态
            state_mu.append(z_t_mean)
            state_logsigma.append(z_t_logsigma)
            sampled_state.append(z_t)

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

    def forward_prediction(self, external_input_seq, n_traj=16, max_prob=False, memory_state=None):
        """
        给定模型记忆memory_state和系统外部输入序列(len,batch_size, input_size)估计系统未来输出的分布，
        并更新memory_state

        Args:
            external_input_seq:
            max_prob: 从预测的系统输出分布中进行采样的方法：max_prob为True，按最大概率采样；max_prob为False，随机采样;
            memory_state:

        Returns:

        """

        with torch.no_grad():

            external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]

            l, batch_size, _ = external_input_seq.size()

            device = external_input_seq.device

            if memory_state is None:
                h, rnn_hidden_state = self.init_h(batch_size, device)
            else:
                h, rnn_hidden_state = memory_state['hn'], memory_state['rnn_hidden']

            predicted_seq_sample = []

            external_input_seq = external_input_seq.repeat(1, n_traj, 1)
            h = h.repeat(n_traj, 1)
            rnn_hidden_state = rnn_hidden_state.repeat(n_traj, 1)

            for i in range(l):
                # 估计每一时刻t下，z的后验分布
                z_t_mean, z_t_logsigma = self.prior_gauss(
                    h
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

                external_input_seq_embed_i = self.process_u(external_input_seq[i])
                h, rnn_hidden_state = self.GRU_update(torch.cat([
                    external_input_seq_embed_i,
                    self.process_z(z_t),
                ], dim=-1), rnn_hidden_state
                )

                h = self.diffeq_solver(h, torch.stack([torch.zeros_like(dt[i, :, 0]), dt[i, :, 0]]))[-1]
                rnn_hidden_state = self.update_rnn_hidden_state(h, rnn_hidden_state)

                observations_sample = split_first_dim(x_t, (n_traj, batch_size))
                observations_sample = observations_sample.permute(1, 0, 2)
                predicted_seq_sample.append(observations_sample)

            predicted_seq_sample = torch.stack(predicted_seq_sample, dim=0)
            predicted_seq = torch.mean(predicted_seq_sample, dim=2)
            predicted_dist = MultivariateNormal(
                predicted_seq_sample.mean(dim=2), torch.diag_embed(predicted_seq_sample.var(dim=2))
                # 此处如何生成分布(如何提取均值和方差)
            )

            outputs = {
                'predicted_dist': predicted_dist,
                'predicted_seq': predicted_seq,
                'predicted_seq_sample': predicted_seq_sample
            }
            h = h[:batch_size]
            rnn_hidden_state = rnn_hidden_state[:batch_size]
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

        h_seq = h_seq[:-1]
        rnn_hidden_state_seq = rnn_hidden_state_seq[:-1]

        # 此处为latent overshooting，d为over_shooting的距离，这部分代码思维强度有点深，需要多看几遍。
        for d in range(D):
            length = h_seq.size()[0]

            # 预测的隐变量先验分布

            prior_z_t_seq_mean, prior_z_t_seq_logsigma = self.prior_gauss(
                h_seq
            )
            # 计算loss中的kl散度项
            kl_sum += multivariate_normal_kl_loss(
                state_mu[-length:].detach() if d > 0 else state_mu[-length],
                logsigma2cov(state_logsigma[-length:].detach()) if d > 0 else logsigma2cov(state_logsigma[-length]),
                prior_z_t_seq_mean,
                logsigma2cov(prior_z_t_seq_logsigma)
            )

            # 利用reparameterization trick 采样z
            z_t_seq = normal_differential_sample(
                MultivariateNormal(prior_z_t_seq_mean, logsigma2cov(prior_z_t_seq_logsigma))
            )
            z_t_seq_embed = self.process_z(z_t_seq)

            h_seq, rnn_hidden_state_seq = self.GRU_update(
                merge_first_two_dims(
                    torch.cat([
                        external_input_seq_embed[-length:],
                        z_t_seq_embed
                    ], dim=-1)
                ),
                merge_first_two_dims(rnn_hidden_state_seq)
            )

            h_seq = split_first_dim(
                self.diffeq_solver(h_seq,
                                   torch.stack([
                                       merge_first_two_dims(torch.zeros_like(dt[-length:, :, 0])),
                                       merge_first_two_dims(dt[-length:, :, 0]),
                                   ])
                                   )[-1], (length, batch_size)
            )[:-1]

            rnn_hidden_state_seq = split_first_dim(rnn_hidden_state_seq, (length, batch_size))[:-1]
            rnn_hidden_state_seq = self.update_rnn_hidden_state(h_seq, rnn_hidden_state_seq)

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
            'loss': (kl_sum - generative_likelihood)/batch_size/l,
            'kl_loss': kl_sum/batch_size/l,
            'likelihood_loss': -generative_likelihood/batch_size/l
        }


    def init_h(self, batch_size, device):

        rnn_hidden_state =  torch.zeros(batch_size, 2 * self.k, device=device)
        h = rnn_hidden_state[ ..., :self.k]
        return h, rnn_hidden_state

    def update_rnn_hidden_state(self, h, rnn_hidden_state):
        return torch.cat([h, rnn_hidden_state[..., -self.k:]], dim=-1)

    def GRU_update(self, x, rnn_hidden_state):
        rnn_hidden_state = self.gru_cell(x, rnn_hidden_state)
        return rnn_hidden_state[..., :self.k], rnn_hidden_state

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
