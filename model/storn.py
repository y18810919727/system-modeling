#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from model.common import DBlock, PreProcess, DBlock_Relu, DBlock_Relu_Mini
from model.common import DiagMultivariateNormal as MultivariateNormal
from common import logsigma2cov, split_first_dim, merge_first_two_dims
from model.func import normal_differential_sample, multivariate_normal_kl_loss, zeros_like_with_shape

import torch.distributions as tdist
"""implementation of the STOchastich Recurent Neural network (STORN) from https://arxiv.org/abs/1411.7610 using
unimodal isotropic gaussian distributions for inference, prior, and generating models."""


class STORN(nn.Module):

    def __init__(self, input_size, state_size, observations_size, k=16, num_layers=1):

        super(STORN, self).__init__()

        self.k = k
        self.observations_size = observations_size
        self.state_size = state_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.rnn_inf = torch.nn.GRU(k, k, num_layers)
        self.rnn_gen = torch.nn.GRU(2*k, k, num_layers)

        self.process_u = PreProcess(input_size, k)
        self.process_x = PreProcess(observations_size, k)
        self.process_z = PreProcess(state_size, k)

        self.posterior_gauss = DBlock_Relu(k, k, state_size)
        self.prior_gauss = DBlock_Relu_Mini(k, k, state_size)
        self.decoder = DBlock_Relu_Mini(k, k, observations_size)

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):

        d0 = None if memory_state is None else memory_state['dn']

        l, batch_size, _ = external_input_seq.size()

        external_input_seq_embed = self.process_u(external_input_seq)
        observations_seq_embed = self.process_x(observations_seq)

        # inference recurrence: d_t, x_t -> d_t+1
        d_seq, dn = self.rnn_inf(observations_seq_embed, d0)

        hn = zeros_like_with_shape(observations_seq, (batch_size, self.k)
                                   ) if memory_state is None else memory_state['hn']

        state_mu = []
        state_logsigma = []
        sampled_state = []
        h_seq = [hn]
        for t in range(l):
            # 后验网络  q(z_t | dt)
            z_t_mean, z_t_logsigma = self.posterior_gauss(
                d_seq[t]
            )

            z_t = normal_differential_sample(
                MultivariateNormal(z_t_mean, logsigma2cov(z_t_logsigma))
            )
            # z_t_dist = tdist.Normal(z_t_mean, z_t_logsigma.exp().sqrt())
            # z_t = tdist.Normal.rsample(z_t_dist)

            # rnn_gen网络更新h_t: u_t+1, z_t, h_t ->h_t+1
            output, _ = self.rnn_gen(
                torch.cat([self.process_z(z_t), external_input_seq_embed[t]], dim=-1).unsqueeze(dim=0),
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
        predicted_mu = []
        predicted_sigma = []

        with torch.no_grad():
            prior_z_t_mean = zeros_like_with_shape(external_input_seq, (batch_size * n_traj, self.state_size))
            prior_z_t_logsigma = zeros_like_with_shape(external_input_seq, (batch_size * n_traj, self.state_size))
            hn = hn.repeat(n_traj, 1)
            for t in range(l):

                # encoder: z~N(0,1)
                z_t_dist = MultivariateNormal(prior_z_t_mean, logsigma2cov(prior_z_t_logsigma))
                z_t = normal_differential_sample(z_t_dist)
                # z_t_dist = tdist.Normal(prior_z_t_mean, prior_z_t_logsigma.exp().sqrt())
                # z_t = tdist.Normal.rsample(z_t_dist)

                # decoder: h_t -> x_t
                observations_dist = self.decode_observation({'h_seq': hn},
                                                            mode='dist')
                # mu, sigma = self.decoder(hn)
                # # observations_dist = MultivariateNormal(
                # #     mu, logsigma2cov(sigma)
                # # )
                # observations_dist = tdist.Normal(
                #     mu, sigma.exp().sqrt()
                # )
                # mu = split_first_dim(
                #     mu,
                #     (n_traj, batch_size)
                # )
                # mu = mu.permute(1, 0, 2)
                # sigma = split_first_dim(
                #     sigma,
                #     (n_traj, batch_size)
                # )
                # sigma = sigma.permute(1, 0, 2)

                # observations_sample = split_first_dim(
                #     tdist.Normal.rsample(observations_dist),
                #     (n_traj, batch_size)
                # )
                observations_sample = split_first_dim(
                    normal_differential_sample(observations_dist),
                    (n_traj, batch_size)
                )
                observations_sample = observations_sample.permute(1, 0, 2)
                predicted_seq_sample.append(observations_sample)

                # predicted_mu.append(mu)
                # predicted_sigma.append(sigma)

                external_input_seq_embed = self.process_u(external_input_seq[t]).repeat(n_traj, 1)
                # rnn_gen网络更新h_t: u_t+1, z_t, h_t ->h_t+1
                output, _ = self.rnn_gen(
                    torch.cat([self.process_z(z_t), external_input_seq_embed], dim=-1).unsqueeze(dim=0),
                    hn.unsqueeze(dim=0))
                hn = output[0]

        predicted_seq_sample = torch.stack(predicted_seq_sample, dim=0)
        predicted_seq = torch.mean(predicted_seq_sample, dim=2)
        predicted_dist = MultivariateNormal(
            predicted_seq_sample.mean(dim=2), torch.diag_embed(predicted_seq_sample.var(dim=2))
        )
        # predicted_mu = torch.stack(predicted_mu, dim=0)
        # predicted_sigma = torch.stack(predicted_sigma, dim=0)
        # predicted_seq = torch.mean(predicted_mu, dim=2)
        # predicted_dist = MultivariateNormal(
        #     predicted_seq, logsigma2cov(torch.mean(predicted_sigma, dim=2))
        #
        # )

        outputs = {
            'predicted_seq_sample': predicted_seq_sample,
            'predicted_dist': predicted_dist,
            'predicted_seq': predicted_seq
        }
        return outputs, {'hn': hn}

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)

        l, batch_size, _ = observations_seq.shape

        # sampled_state = outputs['sampled_state']
        h_seq = outputs['h_seq']
        state_mu = outputs['state_mu']
        state_logsigma = outputs['state_logsigma']

        kl_sum = 0

        prior_z_t_seq_mean = zeros_like_with_shape(external_input_seq, (batch_size, self.state_size))
        prior_z_t_seq_logsigma = zeros_like_with_shape(external_input_seq, (batch_size, self.state_size))

        # kl    z~N(0,1)
        kl_sum += multivariate_normal_kl_loss(
            state_mu,
            logsigma2cov(state_logsigma),
            prior_z_t_seq_mean,
            logsigma2cov(prior_z_t_seq_logsigma)
        )

        # decoder : h_t -> x_t
        observations_normal_dist = self.decode_observation(
            {'h_seq': h_seq},
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
                outputs['h_seq']
        )
        observations_normal_dist = MultivariateNormal(
            mean, logsigma2cov(logsigma)
        )
        # observations_normal_dist = tdist.Normal(
        #     mean, logsigma.exp().sqrt()
        # )

        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return observations_normal_dist.sample()
            # return tdist.Normal.rsample(observations_normal_dist)

