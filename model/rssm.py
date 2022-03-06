
#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from model.common import DBlock, PreProcess
from model.common import DiagMultivariateNormal as MultivariateNormal
from common import logsigma2cov
from model.func import normal_differential_sample
"""Deterministic and stochastic state model.

  The stochastic latent is computed from the hidden state at the same time
  step. If an observation is present, the posterior latent is compute from both
  the hidden state and the observation.

  Prior:    Posterior:

  (a)       (a)
     \         \
      v         v
  [h]->[h]  [h]->[h]
      ^ |       ^ :
     /  v      /  v
  (s)  (s)  (s)  (s)
                  ^
                  :
                 (o)
  """


class RSSM(nn.Module):
    """
        This class includes multiple components
        Deterministic state model: h_t+1 = f(h_t, s_t, a_t)
        Stochastic state model (prior): p(s_t+1 | h_t+1)
        State posterior: q(s_t | h_t, o_t)
        NOTE: actually, this class takes embedded observation by Encoder class
        min_stddev is added to stddev same as original implementation
        Activation function for this class is F.relu same as original implementation
    """
    def __init__(self, input_size, state_size, observation_size, rnn_hidden_size, min_stddev=0.1, act=F.tanh, k=16,
                 num_layers=1, D=1):

        super(RSSM, self).__init__()

        # self.state_dim = state_dim
        # self.action_dim = action_dim
        # self.rnn_hidden_dim = rnn_hidden_dim
        # self.fc_state_action = nn.Linear(state_dim + action_dim, hidden_dim)
        # self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, hidden_dim)
        # self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        # self.fc_state_stddev_prior = nn.Linear(hidden_dim, state_dim)
        # self.fc_rnn_hidden_embedded_obs = nn.Linear(rnn_hidden_dim + 1024, hidden_dim)  #??
        # self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        # self.fc_state_stddev_posterior = nn.Linear(hidden_dim, state_dim)
        # self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        # self._min_stddev = min_stddev

        self.k = k
        self.observations_size = observation_size
        self.state_size = state_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.D = D

        self.rnn = torch.nn.GRUCell(k, k)

        self.process_u = PreProcess(input_size, k)
        self.process_x = PreProcess(observation_size, k)

        self.posterior_gauss = DBlock(state_size+k, 2*k, state_size)
        self.prior_gauss = DBlock(state_size+k, 2*k, state_size)
        self.decoder = DBlock(state_size+k, 2*k, observation_size)
        self.act = act

    def forward_posterior(self, external_input_seq, observation_seq, memory_state=None):
        h0 = None if memory_state is None else memory_state['hn']

        l, batch_size, _ = external_input_seq.size()




        external_input_seq_embed = self.process_u(external_input_seq)
        h_seq, hn = self.rnn(external_input_seq_embed, h0)

        observation_seq_embed = self.process_x(observation_seq)

        s_t_minus_one = torch.zeros((batch_size, self.state_size), device=external_input_seq.device
                                    ) if memory_state is None else memory_state['sn']
        state_mu = []
        state_logsigma = []
        sampled_state = []
        for t in range(l):
            s_t_mean, s_t_logsigma = self.posterior_gauss(
                torch.cat([s_t_minus_one, h_seq[t], observation_seq_embed[t]], dim=-1)
            )

            s_t_minus_one = normal_differential_sample(
                MultivariateNormal(s_t_mean, logsigma2cov(s_t_logsigma))
            )

            state_mu.append(s_t_mean)
            state_logsigma.append(s_t_logsigma)
            sampled_state.append(s_t_minus_one)

        state_mu = torch.stack(state_mu, dim=0)
        state_logsigma = torch.stack(state_logsigma, dim=0)
        sampled_state = torch.stack(sampled_state, dim=0)

        outputs = {
            'state_mu': state_mu,
            'state_logsigma': state_logsigma,
            'sampled_state': sampled_state,
            'h_seq': h_seq,
            'external_input_seq_embed': external_input_seq_embed,
        }

        return outputs, {'hn': hn, 'sn': s_t_minus_one}

    # def forward(self, state, action, rnn_hidden, embedded_next_obs):
    #     """
    #     h_t+1 = f(h_t, s_t, a_t)
    #     Return prior p(s_t+1 | h_t+1) and posterior p(s_t+1 | h_t+1, o_t+1)
    #     for model training
    #     """
    #     next_state_prior, rnn_hidden = self.prior(state, action, rnn_hidden)
    #     next_state_posterior = self.posterior(rnn_hidden,embedded_next_obs)
    #     return next_state_prior, next_state_posterior, rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs):
        """
        Compute posterior q(s_t | h_t, o_t)
        """
        hidden = self.act(self.fc_rnn_hidden_embedded_obs(
            torch.cat([rnn_hidden, embedded_obs], dim=1)))
        mean = self.fc_state_mean_posterior(hidden)
        stddev = F.softplus(self.fc_state_stddev_posterior(hidden)) + self._min_stddev
        return Normal(mean, stddev)

    def prior(self, state, action, rnn_hidden):
        """
        h_t+1 = f(h_t, s_t, a_t)
        Compute prior p(s_t+1 | h_t+1)
        """
        hidden = self.act(self.fc_state_action(torch.cat([state, action], dim=1)))
        rnn_hidden = self.rnn(hidden, rnn_hidden)
        hidden = self.act(self.fc_rnn_hidden(rnn_hidden))

        mean = self.fc_state_mean_prior(hidden)
        stddev = F.softplus(self.fc_state_stddev_prior(hidden))+ self._min_stddev
        return  Normal(mean, stddev), rnn_hidden

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)
        D = self.D if self.training else 1
        l, batch_size, _ = observations_seq.shape

        sampled_state = outputs['sampled_state']
        h_seq = outputs['h_seq']
        state_mu = outputs['state_mu']
        state_logsigma = outputs['state_logsigma']

        s_t_minus_one = torch.zeros(
            (batch_size, self.state_size), device=external_input_seq.device
        ) if memory_state is None else memory_state['sn']

        s_t_minus_one_seq = torch.cat([z_t_minus_one.unsqueeze(0), sampled_state[:-1]], dim=0)

        kl_sum = 0

        predicted_state_sampled = s_t_minus_one_seq   #[l, batch_size, state_size]
        for step in range(D):
            length = predicted_state_sampled.size()[]
            prior_s_t_seq_mean, prior_s_t_seq_logsigma = self.



        states = torch.zeros(
            args.chunk_length, args.batch_size, args.state_dim, device=device)
        rnn_hiddens = torch.zeros(
            args.chunk_length, args.batch_size, args.rnn_hidden_dim, device=device)

        # initialize state and rnn hidden state with 0 vector
        state = torch.zeros(args.batch_size, args.state_dim, device=device)
        rnn_hidden = torch.zeros(args.batch_size, args.rnn_hidden_dim, device=device)

        # compute state and rnn hidden sequences and kl loss
        for step in range(D):
            next_state_prior, next_state_posterior, rnn_hidden = \
                rssm(state, actions[l], rnn_hidden, embedded_observations[l + 1])
            state = next_state_posterior.rsample()
            states[l + 1] = state
            rnn_hiddens[l + 1] = rnn_hidden
            kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
            kl_sum += kl.clamp(min=args.free_nats).mean()
        kl_sum /= D

        observations_normal_dist = self.decode_observation(
            {'sampled_state':sampled_state, 'h_seq':h_seq},
        mode='dist')

        generative_likelihood = torch.sum(observations_normal_dist.log_prob(observations_seq))

        return {
            'loss':(kl_sum - generative_likelihood)/batch_size,
            'kl_loss':kl_sum/batch_size,
            'likelihood_loss':-generative_likelihood/batch_size
        }



    def decode_observation(self, outputs, mode='sample'):
        """
        p(o_t | s_t, h_t)
        Observation model to reconstruct image observation (3, 64, 64)
        from state and rnn hidden state
        """
        mean, logsigma = self.decoder(
                torch.cat([outputs['sampled_state'], outputs['d_seq']], dim=-1)
        )
        observations_normal_dist = MultivariateNormal(
            mean, logsigma2cov(logsigma)
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return observations_normal_dist.sample()













# class RewardModel(nn.Module):
#     """
#     p(r_t | s_t, h_t)
#     Reward model to predict reward from state and rnn hidden state
#     """
#     def __init__(self, state_dim, rnn_hidden_dim, hidden_dim = 300, act = F.relu):
#         super(RewardModel, self).__init__()
#         self.fc1 = nn.Linear(state_dim, rnn_hidden_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, 1)
#         self.act = act
#
#     def forward(self, state, rnn_hidden):
#         hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
#         hidden = self.act(self.fc2(hidden))
#         hidden = self.act(self.fc3(hidden))
#         reward = self.fc4(hidden)
#         return reward
