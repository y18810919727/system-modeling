#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn

from model.vaeakf_v1 import VAEAKF as VAEAKF_V1


class VAEAKF(VAEAKF_V1):

    def call_loss(self):

        l, bs, _  = self.observations_seq.shape

        # region calculating kl divergence
        prior_mu = torch.zeros((l, bs, self.state_size)).to(self.observations_seq.device)
        prior_cov = torch.zeros((l, bs, self.state_size)).to(self.observations_seq.device)

        prior_mu[0]=self.initial_prior_mu
        prior_mu[1:] = torch.nn.functional.linear(
            self.state_mu[0:-1],self.estimate_A) + torch.nn.functional.linear(self.external_input_seq[:-1], self.estimate_B)
        prior_cov[0] = self.initial_prior_sigma**2
        prior_cov[1:] = torch.nn.functional.linear(
            torch.exp(self.state_logsigma[:-1])**2, self.estimate_A
        ) + torch.exp(self.estimate_logsigma)**2

        kl = self.multivariateNormal_kl_loss(prior_mu, prior_cov, self.state_mu, torch.exp(self.state_logsigma)**2)
        # endregion

        # region calculation generative probability p(x|z)

        """
        z(t) = H*x(t) + v(t) is also a gaussian system
        if
            x(t) \sim N(mu, cov)
            v(t) \sim N(0, delta^2)
        then
            z(t) \sim N(H*mu, H*cov*H^T+ + delta^2)
        """

        observation_mu = torch.nn.functional.linear(self.state_mu, self.estimate_H)
        observation_cov = self.estimate_H.matmul(
            torch.diag_embed(torch.exp(self.state_logsigma)**2)
        ).matmul(self.estimate_H.t()) + torch.diag_embed(torch.exp(self.estimate_logdelta)**2)

        observation_normal_dist = torch.distributions.MultivariateNormal(observation_mu, observation_cov)
        generative_likelihood = torch.sum(observation_normal_dist.log_prob(self.observations_seq))

        # endregion

        # maximun -kl + generative_likelihood
        return (kl - generative_likelihood)/bs
