#!/usr/bin/python
# -*- coding:utf8 -*-

import torch
from torch import nn
from model.ct_model import ODEFunc
from model.ct_model import DiffeqSolver
from model.ct_model import linspace_vector
from model.common import PreProcess, DBlock

import torch.nn.functional as F
from model.common import DiagMultivariateNormal as MultivariateNormal
from common import logsigma2cov, split_first_dim, merge_first_two_dims
from model.common import DiagMultivariateNormal as MultivariateNormal, MLP
from model.func import zeros_like_with_shape, normal_differential_sample


class ODE_RNN(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 ode_hidden_dim=50,
                 ode_solver='euler',
                 ode_num_layers=1,
                 rtol=1e-3,
                 atol=1e-4
                 ):

        super(ODE_RNN, self).__init__()
        input_size -= 1
        self.input_size, self.output_size, self.hidden_size, self.ode_hidden_dim, self.ode_solver = \
            input_size, output_size, hidden_size, ode_hidden_dim, ode_solver

        self.rnn_cell = nn.GRUCell(
            hidden_size*2,
            hidden_size*2
        )

        self.process_u = PreProcess(input_size, hidden_size)
        self.process_x = PreProcess(output_size, hidden_size)
        # self.Ly = nn.Linear(2*hidden_size, output_size)
        self.Ly_gauss = DBlock(2*hidden_size, 2*hidden_size, output_size)

        self.gradient_net = MLP(hidden_size, ode_hidden_dim, hidden_size, num_mlp_layers=ode_num_layers)
        ode_func = ODEFunc(
            ode_net=self.gradient_net,
            inputs=None,
            ode_type='normal'
        )
        self.diffeq_solver = DiffeqSolver(hidden_size, ode_func, ode_solver,
                                          odeint_rtol=rtol, odeint_atol=atol)

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):

        external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]

        l, batch_size, _ = external_input_seq.size()

        external_input_seq_embed = self.process_u(external_input_seq)
        observations_seq_embed = self.process_x(observations_seq)

        hn = torch.zeros((batch_size, 2*self.hidden_size), device=external_input_seq.device) if memory_state is None else memory_state['hn']

        state_mu = []
        state_logsigma = []
        sample_state = []
        h_seq = [hn]
        for t in range(l):

            x_t_mean, x_t_logsigma = self.Ly_gauss(hn)  # 先验和后验这边是否用同一个高斯网络？之前是同一个Ly

            x_t = normal_differential_sample(
                MultivariateNormal(x_t_mean, logsigma2cov(x_t_logsigma))
            )

            # encoder网络更新h_t: u_t, x_t, h_t -> h_t+1
            hn = self.rnn_cell(
                torch.cat([external_input_seq_embed[t], observations_seq_embed[t]], dim=-1),
                hn
            )
            h, c = hn[..., :self.hidden_size], hn[..., -self.hidden_size:]
            h = self.diffeq_solver(h, torch.stack([torch.zeros_like(dt[t, :, 0]), dt[t, :, 0]]))[-1]
            hn = torch.cat([h, c], dim=-1)

            state_mu.append(x_t_mean)
            state_logsigma.append(x_t_logsigma)
            sample_state.append(x_t)
            h_seq.append(hn)

        state_mu = torch.stack(state_mu, dim=0)
        state_logsigma = torch.stack(state_logsigma, dim=0)
        sample_state = torch.stack(sample_state, dim=0)
        h_seq = h_seq[:-1]
        h_seq = torch.stack(h_seq, dim=0)
        # x_seq = self.Ly(h_seq)
        # x_seq_mean, = self.Ly_gauss(h_seq)

        outputs = {
            'state_mu': state_mu,
            'state_logsigma': state_logsigma,
            'h_seq': h_seq,
            'predicted_seq': sample_state,
            'observations_seq_embed': observations_seq_embed,
        }

        return outputs, {'hn': hn}

    def forward_prediction(self, external_input_seq, n_traj=1, memory_state=None):
        l, batch_size, _ = external_input_seq.size()
        external_input_seq = external_input_seq.repeat(1, n_traj, 1)
        external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]

        # input_n_traj = n_traj
        # n_traj = 1

        external_input_seq_embed = self.process_u(external_input_seq)

        hn = torch.zeros((batch_size, 2*self.hidden_size), device=external_input_seq.device) if memory_state is None else memory_state['hn']

        predicted_seq_sample = []

        with torch.no_grad():
            hn = hn.repeat(n_traj, 1)
            for t in range(l):

                # decoder: h_t -> x_t+1
                # x_t = self.Ly(hn)
                x_t_mean, x_t_logsigma = self.Ly_gauss(hn)

                x_t = normal_differential_sample(
                    MultivariateNormal(x_t_mean, logsigma2cov(x_t_logsigma))
                )
                # rnn网络更新h_t: u_t+1, x_t+1, h_t ->h_t+1
                # output, _ = self.rnn_encoder(
                #     torch.cat([external_input_seq_embed[t], self.process_x(x_t)], dim=-1),
                #     hn)
                # hn = output[0]

                hn = self.rnn_cell(
                    torch.cat([external_input_seq_embed[t], self.process_x(x_t)], dim=-1),
                    hn
                )
                h, c = hn[..., :self.hidden_size], hn[..., -self.hidden_size:]
                h = self.diffeq_solver(h, torch.stack([torch.zeros_like(dt[t, :, 0]), dt[t, :, 0]]))[-1]
                hn = torch.cat([h, c], dim=-1)

                observation = split_first_dim(x_t, (n_traj, batch_size))
                observation = observation.permute(1, 0, 2)
                predicted_seq_sample.append(observation)

        predicted_seq_sample = torch.stack(predicted_seq_sample, dim=0)
        predicted_seq = torch.mean(predicted_seq_sample, dim=2)
        predicted_dist = MultivariateNormal(
            predicted_seq_sample.mean(dim=2), torch.diag_embed(predicted_seq_sample.var(dim=2))
        )

        outputs = {
            'predicted_seq_sample': predicted_seq_sample,
            'predicted_dist': predicted_dist,
            'predicted_seq': predicted_seq
        }
        return outputs, {'hn': hn}

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        # outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)
        l, batch_size, _ = observations_seq.shape

        train_pred_len = int(len(external_input_seq) / 2)
        historical_input = external_input_seq[:train_pred_len]
        historical_ob = observations_seq[:train_pred_len]
        future_input = external_input_seq[train_pred_len:]
        future_ob = observations_seq[train_pred_len:]
        all_ob = observations_seq

        outputs, memory_state = self.forward_posterior(historical_input, historical_ob, memory_state)
        observations_normal_dist = self.decode_observation(outputs, mode='dist')
        generative_likelihood = torch.sum(observations_normal_dist.log_prob(historical_ob))
        # TODO:KL_loss加不加?

        # reconstructing_x_seq = outputs['predicted_seq']
        #
        # outputs, memory_state = self.forward_prediction(future_input, memory_state=memory_state)
        # predicted_x_seq = outputs['predicted_seq']
        # all_predicted_seq = torch.cat([reconstructing_x_seq, predicted_x_seq], dim=0)


        return {
            'loss': -generative_likelihood/batch_size/l,  # TODO:改成观测数据的似然，类似于odeRssm的likehoodloss
            'kl_loss': 0,
            'likelihood_loss': -generative_likelihood/batch_size/l
        }

    def decode_observation(self, outputs, mode='sample'):
        """

              Args:
                  outputs:
                  mode: dist or sample

              Returns:
                  model为sample时，从分布采样(len,batch_size,observation)
                  为dist时，直接返回分布对象torch.distributions.MultivariateNormal

              方法调用时不会给额外的输入参数，需在每次forward_prediction和forward_posterior之后将解码所需的信息存储在self里
              """
        mean, logsigma = outputs['state_mu'], outputs['state_logsigma']
        observations_normal_dist = MultivariateNormal(
            mean, logsigma2cov(logsigma)
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return observations_normal_dist.sample()

