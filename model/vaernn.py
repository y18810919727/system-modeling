#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from model.common import DBlock, PreProcess
from model.common import DiagMultivariateNormal as MultivariateNormal
from common import logsigma2cov, split_first_dim, merge_first_two_dims
from model.func import normal_differential_sample, multivariate_normal_kl_loss, zeros_like_with_shape

"""implementation of the STOchastich Recurent Neural network (STORN) from https://arxiv.org/abs/1411.7610 using
unimodal isotropic gaussian distributions for inference, prior, and generating models."""


class VAERNN(nn.Module):

    def __init__(self, input_size, state_size, observations_size, k=16, num_layers=1, D=1):

        super(VAERNN, self).__init__()

        self.k = k
        self.observations_size = observations_size
        self.state_size = state_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.D = D

        self.rnn = torch.nn.GRU(k, k, num_layers)

        self.process_u = PreProcess(input_size, k)
        self.process_x = PreProcess(observations_size, k)
        self.process_z = PreProcess(state_size, k)

        self.posterior_gauss = DBlock(2*k, 2*k, state_size)
        self.prior_gauss = DBlock(k, 2*k, state_size)
        self.decoder = DBlock(state_size, 2*k, observations_size)

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):

        # d0 = None if memory_state is None else memory_state['dn']

        l, batch_size, _ = external_input_seq.size()

        external_input_seq_embed = self.process_u(external_input_seq)
        observations_seq_embed = self.process_x(observations_seq)

        # inference recurrence: d_t, x_t -> d_t+1
        # d_seq, dn = self.rnn_inf(observations_seq_embed, d0)

        hn = zeros_like_with_shape(observations_seq, (batch_size, self.k)
                                   ) if memory_state is None else memory_state['hn']
        # z_t = torch.zeros((batch_size, self.state_size),
        #                             device=external_input_seq.device) if memory_state is None else memory_state['zn']
        state_mu = []
        state_logsigma = []
        sampled_state = []
        h_seq = [hn]
        for t in range(l):
            # 后验网络  q(z_t | x_t, h_t)
            z_t_mean, z_t_logsigma = self.posterior_gauss(
                torch.cat([observations_seq_embed[t], hn], dim=-1)
            )

            z_t = normal_differential_sample(
                MultivariateNormal(z_t_mean, logsigma2cov(z_t_logsigma))
            )

            # rnn_gen网络更新h_t: u_t+1, h_t ->h_t+1
            output, _ = self.rnn(external_input_seq_embed[t].unsqueeze(dim=0),
                hn.unsqueeze(dim=0))
            hn = output[0]

            state_mu.append(z_t_mean)
            state_logsigma.append(z_t_logsigma)
            sampled_state.append(z_t)
            h_seq.append(hn)

        state_mu = torch.stack(state_mu, dim=0)
        state_logsigma = torch.stack(state_logsigma, dim=0)
        sampled_state = torch.stack(sampled_state, dim=0)
        h_seq = torch.stack(h_seq, dim=0)
        h_seq = h_seq[:-1]

        outputs = {
            'state_mu': state_mu,
            'state_logsigma': state_logsigma,
            'sampled_state': sampled_state,
            'h_seq': h_seq,
            'observations_seq_embed': observations_seq_embed,
        }

        return outputs, {'hn': hn}

    def forward_prediction(self, external_input_seq, n_traj, memory_state=None):

        l, batch_size, _ = external_input_seq.size()

        hn = zeros_like_with_shape(external_input_seq, (batch_size, self.k)
                                   ) if memory_state is None else memory_state['hn']

        predicted_seq_sample = []

        with torch.no_grad():
            hn = hn.repeat(n_traj, 1)
            for t in range(l):

                # 先验网络: h_t -> z_t
                prior_z_t_mean, prior_z_t_logsigma = self.prior_gauss(
                    hn
                )

                z_t_dist = MultivariateNormal(prior_z_t_mean, logsigma2cov(prior_z_t_logsigma))
                z_t = normal_differential_sample(z_t_dist)

                # decoder: z_t -> x_t
                observations_dist = self.decode_observation({'sampled_state': z_t},
                                                            mode='dist')
                observations_sample = split_first_dim(
                    normal_differential_sample(observations_dist),
                    (n_traj, batch_size)
                )
                observations_sample = observations_sample.permute(1, 0, 2)
                predicted_seq_sample.append(observations_sample)
                external_input_seq_embed = self.process_u(external_input_seq[t]).repeat(n_traj, 1)
                # rnn网络更新h_t: u_t+1, h_t ->h_t+1
                output, _ = self.rnn(external_input_seq_embed.unsqueeze(dim=0),
                    hn.unsqueeze(dim=0))
                hn = output[0]

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
        outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)
        D = self.D if self.training else 1
        l, batch_size, _ = observations_seq.shape

        external_input_seq_embed = self.process_u(external_input_seq)

        sampled_state = outputs['sampled_state']
        h_seq = outputs['h_seq']
        state_mu = outputs['state_mu']
        state_logsigma = outputs['state_logsigma']

        kl_sum = 0

        predicted_h = h_seq

        for step in range(D):
            length = predicted_h.size()[0]

            # 先验网络预测z_t  p(z_t | h_t)   for KL loss
            prior_z_t_seq_mean, prior_z_t_seq_logsigma = self.prior_gauss(
                predicted_h
            )

            kl_sum += multivariate_normal_kl_loss(
                state_mu[-length:].detach() if step > 0 else state_mu[-length],
                logsigma2cov(state_logsigma[-length:].detach()) if step > 0 else logsigma2cov(state_logsigma[-length]),
                prior_z_t_seq_mean,
                logsigma2cov(prior_z_t_seq_logsigma)
            )

            # recurrence: u_t+1, h_t -> h_t+1    # u_t, h_t-1 -> h_t  u_t更新的是x_t+1
            output, _ = self.rnn(
                merge_first_two_dims(external_input_seq_embed[-length:]).unsqueeze(dim=0),
                merge_first_two_dims(predicted_h).unsqueeze(dim=0)
            )

            predicted_h = split_first_dim(output[0], (length, batch_size))
            predicted_h = predicted_h[:-1]

        kl_sum = kl_sum/D

        # decoder : z_t -> x_t
        observations_normal_dist = self.decode_observation(
            {'sampled_state': sampled_state},
            mode='dist')

        generative_likelihood = torch.sum(observations_normal_dist.log_prob(observations_seq))

        return {
            'loss': (kl_sum - generative_likelihood)/batch_size/l,
            'kl_loss': kl_sum/batch_size/l,
            'likelihood_loss': -generative_likelihood/batch_size/l
        }

    def decode_observation(self, outputs, mode='sample'):
        """
        p(o_t | s_t, h_t)
        from state and rnn hidden state
        """
        mean, logsigma = self.decoder(
                outputs['sampled_state']
        )
        observations_normal_dist = MultivariateNormal(
            mean, logsigma2cov(logsigma)
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return observations_normal_dist.sample()

