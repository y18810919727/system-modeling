#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from model.common import DBlock, PreProcess, MLP
from model.common import DiagMultivariateNormal as MultivariateNormal
from common import logsigma2cov, split_first_dim, merge_first_two_dims
from model.func import normal_differential_sample, multivariate_normal_kl_loss, zeros_like_with_shape


class RNN(nn.Module):

    def __init__(self, input_size, state_size, observations_size, k=16, num_layers=1, train_pred_len=60, ct_time=None, sp=None):

        super(RNN, self).__init__()

        self.k = k
        self.observations_size = observations_size
        self.state_size = state_size
        self.input_size = input_size
        self.num_layers = num_layers
        if ct_time:
            self.train_pred_len = int(train_pred_len * sp)
        else:
            self.train_pred_len = train_pred_len
        self.process_u = PreProcess(input_size, k)
        self.process_x = PreProcess(observations_size, k)
        self.process_z = PreProcess(state_size, k)

        self.rnn_encoder = torch.nn.GRU(2*k, k, num_layers)
        self.decoder = MLP(k, 2*k, observations_size, num_layers)

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):

        l, batch_size, _ = external_input_seq.size()

        external_input_seq_embed = self.process_u(external_input_seq)
        observations_seq_embed = self.process_x(observations_seq)

        hn = zeros_like_with_shape(observations_seq, (batch_size, self.k)
                                   ) if memory_state is None else memory_state['hn']

        h_seq = [hn]
        x_seq = []
        for t in range(l):

            # encoder网络更新h_t: u_t+1, x_t+1, h_t -> h_t+1
            output, _ = self.rnn_encoder(
                torch.cat([external_input_seq_embed[t], observations_seq_embed[t]], dim=-1).unsqueeze(dim=0),
                hn.unsqueeze(dim=0)
            )
            hn = output[0]
            # x重构 for loss
            x_t = self.decoder(hn.unsqueeze(dim=0))[0]

            h_seq.append(hn)
            x_seq.append(x_t)

        h_seq = torch.stack(h_seq, dim=0)
        h_seq = h_seq[:-1]
        x_seq = torch.stack(x_seq, dim=0)

        outputs = {
            'state_mu': observations_seq,
            'state_logsigma': -torch.ones_like(observations_seq) * float('inf'),
            'h_seq': h_seq,
            'x_seq': x_seq,
            'observations_seq_embed': observations_seq_embed,
        }

        return outputs, {'hn': hn}

    def forward_prediction(self, external_input_seq, n_traj=1, memory_state=None):

        n_traj = 1
        l, batch_size, _ = external_input_seq.size()

        external_input_seq_embed = self.process_u(external_input_seq)
        hn = zeros_like_with_shape(external_input_seq, (batch_size, self.k)
                                   ) if memory_state is None else memory_state['hn']

        predicted_seq_sample = []
        for t in range(l):

            # decoder: h_t -> x_t+1
            x_t = self.decoder(hn.unsqueeze(dim=0))[0]
            # rnn网络更新h_t: u_t+1, x_t+1, h_t ->h_t+1
            output, _ = self.rnn_encoder(
                torch.cat([external_input_seq_embed[t], self.process_x(x_t)], dim=-1).unsqueeze(dim=0),
                hn.unsqueeze(dim=0))
            hn = output[0]
            observation_sample = split_first_dim(x_t, (n_traj, batch_size))
            observation_sample = observation_sample.permute(1, 0, 2)
            predicted_seq_sample.append(observation_sample)

        predicted_seq_sample = torch.stack(predicted_seq_sample, dim=0)
        predicted_seq = torch.mean(predicted_seq_sample, dim=2)
        # n_traj = 1 predicted_seq_sample.var(dim=2) = torch.zero()
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
        # l, batch_size, _ = observations_seq.shape

        historical_input = external_input_seq[:-self.train_pred_len]
        historical_ob = observations_seq[:-self.train_pred_len]
        future_input = external_input_seq[-self.train_pred_len:]
        # future_ob = observations_seq[-self.train_pred_len:]
        future_ob = observations_seq

        outputs, memory_state = self.forward_posterior(historical_input, historical_ob, memory_state)
        posterior_x_seq = outputs['x_seq']

        outputs, memory_state = self.forward_prediction(future_input, memory_state=memory_state)
        predicted_x_seq = outputs['predicted_seq']
        predicted_seq = torch.cat([posterior_x_seq, predicted_x_seq], dim=0)
        return {
            'loss': F.mse_loss(predicted_seq, future_ob),
            'kl_loss': 0,
            'likelihood_loss': 0
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
        if mode == 'dist':
            observations_normal_dist = MultivariateNormal(
                outputs['state_mu'], logsigma2cov(outputs['state_logsigma'])
            )
            return observations_normal_dist
        elif mode == 'sample':
            return outputs['state_mu']

