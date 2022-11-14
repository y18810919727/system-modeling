#!/usr/bin/python
# -*- coding:utf8 -*-

import torch
from torch import nn
from model.common import DBlock, PreProcess
from common import logsigma2cov, split_first_dim, merge_first_two_dims, softplus
from model.common import DiagMultivariateNormal as MultivariateNormal, MLP
from model.func import normal_differential_sample, multivariate_normal_kl_loss, kld_gauss
from model.ct_model import ODEFunc
from model.ct_model.diffeq_solver import solve_diffeq
from model.ct_model.interpolation import interpolate, ConstInterpolation
from einops import rearrange
from torch.nn import GRUCell


class ODERSSM(nn.Module):

    def __init__(self, input_size, state_size, observations_size,
                 ode_solver='dopri5', k=16, D=5, ode_hidden_dim=50, ode_num_layers=1, rtol=1e-3, atol=1e-4,
                 ode_type='normal', detach='all', weight='average', ode_ratio='half', iw_trajs=1,
                 z_in_ode=False, input_interpolation=True):

        super(ODERSSM, self).__init__()

        input_size = input_size - 1  # The last dimension of input variable represents dt
        self.k = k
        self.D = D
        self.observations_size = observations_size
        self.state_size = state_size
        self.input_size = input_size
        self.z_in_ode = z_in_ode
        self.iw_trajs = iw_trajs
        self.input_interpolation = input_interpolation
        if ode_ratio == 'half':
            self.h_size = int(k/2)
        elif ode_ratio == 'all':
            self.h_size = k
        else:
            raise NotImplementedError
        self.gru_cell = GRUCell(2*k, k)

        self.process_x = PreProcess(observations_size, k)
        self.process_u = PreProcess(input_size, k)
        self.process_z = PreProcess(state_size, k)

        self.posterior_gauss = DBlock(2*k, 3*k, state_size)
        self.prior_gauss = DBlock(k, 3*k, state_size)
        self.decoder = DBlock(2*k, 3*k, observations_size)

        input_embed_dim = k
        if self.z_in_ode:
            self.gradient_net = MLP(self.h_size+input_embed_dim+k, ode_hidden_dim, self.h_size, num_mlp_layers=ode_num_layers)
        else:
            self.gradient_net = MLP(self.h_size+input_embed_dim, ode_hidden_dim, self.h_size, num_mlp_layers=ode_num_layers)
        self.ode_type = ode_type
        self.rtol = rtol
        self.atol = atol
        self.ode_solver = ode_solver

        self.weight_initial_hidden_state = None
        self.h_seq = None
        self.external_input_seq_embed = None
        self.observations_seq_embed = None
        self.memory_state = None
        self.rnn_hidden_state_seq = None
        self.detach = detach
        self.weight = weight

    def interpolate(self, u, nu, z, dt):

        if self.input_interpolation:
            y = torch.stack([u, nu], dim=0)
            if self.z_in_ode:
                y = torch.cat([y, z.unsqueeze(0).repeat(2, 1, 1)], dim=-1)
            T = torch.stack([torch.zeros_like(dt), dt], dim=0)
            inputs = interpolate('gp', T, y, batched=True)
        else:
            inputs = ConstInterpolation(
                torch.cat([u, z], dim=-1) if self.z_in_ode else u,
            )
        return inputs

    def ode_prior(self, h, inputs, dt):

        ode_func = ODEFunc(ode_net=self.gradient_net, inputs_interpolation=inputs, ode_type=self.ode_type)
        h = solve_diffeq(
            ode_func, h,
            torch.stack([torch.zeros_like(dt), dt], dim=0),
            self.ode_solver,
            odeint_rtol=self.rtol,
            odeint_atol=self.atol
        )[-1]
        return h

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
                torch.cat([observations_seq_embed[i], rnn_hidden_state], dim=-1)
            )
            # 从分布做采样得到z_t
            z_t = normal_differential_sample(
                MultivariateNormal(z_t_mean, logsigma2cov(z_t_logsigma))
            )
            z_t_embed = self.process_z(z_t)
            h, rnn_hidden_state = self.GRU_update(torch.cat([
                external_input_seq_embed[i],
                z_t_embed,
            ], dim=-1), rnn_hidden_state)

            # dt[i]代表 t[i+1] - t[i]，预测t[i+1]时刻的h

            # h = self.ode_solve(h, dt[i], external_input_seq_embed[i], z_t_embed)
            inputs = self.interpolate(
                external_input_seq_embed[i],
                external_input_seq_embed[min(i+1, l-1)],
                z_t_embed,
                dt[i]
            )

            h = self.ode_prior(h, inputs, dt[i, :, 0])
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

            l, batch_size, _ = external_input_seq.size()
            external_input_seq = external_input_seq.repeat(1, n_traj, 1)
            external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]


            device = external_input_seq.device

            if memory_state is None:
                h, rnn_hidden_state = self.init_h(batch_size, device)
            else:
                h, rnn_hidden_state = memory_state['hn'], memory_state['rnn_hidden']

            predicted_seq_sample = []

            h = h.repeat(n_traj, 1)
            rnn_hidden_state = rnn_hidden_state.repeat(n_traj, 1)

            input_embed = None

            for i in range(l):
                # 估计每一时刻t下，z的后验分布
                z_t_mean, z_t_logsigma = self.prior_gauss(
                    rnn_hidden_state
                )
                z_t = normal_differential_sample(
                    MultivariateNormal(z_t_mean, logsigma2cov(z_t_logsigma))
                )
                z_t_embed = self.process_z(z_t)
                x_t_mean, x_t_logsigma = self.decoder(
                    torch.cat([z_t_embed, rnn_hidden_state], dim=-1)
                )
                x_t = normal_differential_sample(
                    MultivariateNormal(x_t_mean, logsigma2cov(x_t_logsigma))
                )

                external_input_seq_embed_i = self.process_u(external_input_seq[i])

                h, rnn_hidden_state = self.GRU_update(torch.cat([
                    external_input_seq_embed_i,
                    z_t_embed,
                ], dim=-1), rnn_hidden_state
                )

                inputs = self.interpolate(
                    external_input_seq_embed_i,
                    self.process_u(external_input_seq[min(i + 1, l - 1)]),
                    z_t_embed,
                    dt[i]
                )

                h = self.ode_prior(h, inputs, dt[i, :, 0])
                rnn_hidden_state = self.update_rnn_hidden_state(h, rnn_hidden_state)

                observations_sample = split_first_dim(x_t, (n_traj, batch_size))
                observations_sample = observations_sample.permute(1, 0, 2)
                predicted_seq_sample.append(observations_sample.detach())

            predicted_seq_sample = torch.stack(predicted_seq_sample, dim=0)
            predicted_seq = torch.mean(predicted_seq_sample, dim=2)
            predicted_dist = MultivariateNormal(
                predicted_seq, torch.diag_embed(predicted_seq_sample.var(dim=2))
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

        l, batch_size, _ = observations_seq.shape

        # if self.iw_trajs == 1:
        #     return self.call_loss_single(external_input_seq, observations_seq, memory_state=memory_state)
        # else:

        external_input_seq = external_input_seq.repeat(1, self.iw_trajs, 1)
        observations_seq = observations_seq.repeat(1, self.iw_trajs, 1)
        if memory_state is not None:
            for k, v in memory_state.items:
                memory_state[k] = v.repeat(self.iw_trajs, 1)

        n_batch_size = observations_seq.size(1)

        dt = external_input_seq[..., -1:]
        outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)

        h_seq = outputs['h_seq']
        rnn_hidden_state_seq = outputs['rnn_hidden_state_seq']
        state_mu = outputs['state_mu']
        state_logsigma = outputs['state_logsigma']
        external_input_seq_embed = outputs['external_input_seq_embed']

        D = self.D if self.training else 1

        kl_sum = 0

        h_seq = h_seq[:-1]
        rnn_hidden_state_seq = rnn_hidden_state_seq[:-1]

        if self.detach == 'first':
            is_detach = lambda d: d > 0
        elif self.detach == 'half':
            is_detach = lambda d: d > int(D/2)
        elif self.detach == 'none':
            is_detach = lambda _: False
        else:
            is_detach = lambda _: False

        kl_items = []
        for d in range(D):
            length = h_seq.size()[0]

            # 预测的隐变量先验分布

            prior_z_t_seq_mean, prior_z_t_seq_logsigma = self.prior_gauss(
                rnn_hidden_state_seq
            )
            # 计算loss中的kl散度项
            kl = kld_gauss(
                state_mu[-length:].detach() if is_detach(d) else state_mu[-length:],
                torch.log(softplus(state_logsigma[-length:].detach())**2) if is_detach(d) else torch.log(softplus(state_logsigma[-length:])**2),
                prior_z_t_seq_mean,
                torch.log(softplus(prior_z_t_seq_logsigma)**2),
                sum=False
            )
            kl = rearrange(kl, 'l (n bs) s -> l n bs s', n=self.iw_trajs)
            kl = torch.sum(torch.logsumexp(kl, dim=1) - torch.tensor(self.iw_trajs))
            # kl = torch.log(
            #     torch.mean(kl.exp(), dim=1)
            # ).sum()
            kl_items.append(kl)

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

            # if self.z_in_ode:
            #     ode_inputs = torch.cat([external_input_seq_embed[-length:], z_t_seq_embed], dim=-1)
            # else:
            #     ode_inputs = external_input_seq_embed[-length:]

            ode_inputs = self.interpolate(
                merge_first_two_dims(external_input_seq_embed[-length:]),
                merge_first_two_dims(torch.cat([external_input_seq_embed[-length+1:], external_input_seq_embed[-1:]], dim=0)),
                merge_first_two_dims(z_t_seq_embed),
                merge_first_two_dims(dt[-length:])
            )
            ode_func = ODEFunc(
                ode_net=self.gradient_net,
                inputs_interpolation=ode_inputs,
                ode_type=self.ode_type
            )

            h_seq = split_first_dim(
                solve_diffeq(ode_func, h_seq,
                             torch.stack([
                                       merge_first_two_dims(torch.zeros_like(dt[-length:, :, 0])),
                                       merge_first_two_dims(dt[-length:, :, 0]),
                                   ]),
                             self.ode_solver,
                             odeint_rtol=self.rtol,
                             odeint_atol=self.atol)[-1],
                (length, n_batch_size)
            )[:-1]

            rnn_hidden_state_seq = split_first_dim(rnn_hidden_state_seq, (length, n_batch_size))[:-1]
            rnn_hidden_state_seq = self.update_rnn_hidden_state(h_seq, rnn_hidden_state_seq)

        observations_normal_dist = self.decode_observation(outputs, mode='dist')
        # 计算loss的第二部分：观测数据的重构loss
        generative_likelihood = observations_normal_dist.log_prob(observations_seq)
        # generative_likelihood = torch.sum(
        #     rearrange(generative_likelihood, 'l (n bs) -> l n bs', n=self.iw_trajs).exp().mean(dim=1).log()
        # )
        split_generative_likelihood = rearrange(generative_likelihood, 'l (n bs) -> l n bs', n=self.iw_trajs)
        generative_likelihood = torch.sum(torch.logsumexp(split_generative_likelihood, dim=1) - torch.tensor(self.iw_trajs))

        kl_items = torch.stack(kl_items)
        if self.weight == 'average':
            weight = torch.ones_like(kl_items) / D
        elif self.weight == 'decay':
            weight = torch.softmax(torch.arange(0, D).flip(dims=(0,)).float(), dim=-1).to(kl_items.device)
        elif self.weight == 'D_1':
            weight = torch.ones_like(kl_items)
            weight[0] = D
            # weight = torch.softmax(weight, dim=-1)

        kl_sum = (weight*kl_items).sum()

        return {
            'loss': (kl_sum - generative_likelihood)/batch_size/l,
            'kl_loss': kl_sum/batch_size/l,
            'likelihood_loss': -generative_likelihood/batch_size/l
        }

    def init_h(self, batch_size, device):

        rnn_hidden_state = torch.zeros(batch_size, self.k, device=device)
        h = self.extract_h_from_rnn_hidden_state(rnn_hidden_state)
        return h, rnn_hidden_state

    def GRU_update(self, x, rnn_hidden_state):
        rnn_hidden_state = self.gru_cell(x, rnn_hidden_state)
        return self.extract_h_from_rnn_hidden_state(rnn_hidden_state), rnn_hidden_state

    def update_rnn_hidden_state(self, h, rnn_hidden_state):
        return torch.cat([h, rnn_hidden_state[..., self.h_size:]], dim=-1)

    def extract_h_from_rnn_hidden_state(self, rnn_hidden_state):
        return rnn_hidden_state[..., :self.h_size]

    def decode_observation(self, outputs, mode='sample'):
        """

        Args:
            state: with shape (len, batch_size, state_size)
            mode: dist or sample

        Returns:

        """
        mean, logsigma = self.decoder(
            torch.cat([
                self.process_z(outputs['sampled_state']), outputs['rnn_hidden_state_seq'][:-1]
            ], dim=-1)
        )
        observations_normal_dist = MultivariateNormal(
            mean, logsigma2cov(logsigma)
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return observations_normal_dist.sample()
