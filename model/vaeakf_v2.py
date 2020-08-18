#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn

from model.vaeakf_v1 import VAEAKF as VAEAKF_V1
from common import softplus


class VAEAKF(VAEAKF_V1):

    def estimate_generative_logprobability(self):

        observation_mu = torch.nn.functional.linear(self.state_mu, self.estimate_H) + self.estimate_bias
        observation_cov = self.estimate_H.matmul(
            torch.diag_embed(softplus(self.state_logsigma)**2)
        ).matmul(self.estimate_H.t()) + torch.diag_embed(softplus(self.estimate_logdelta)**2)

        observation_normal_dist = torch.distributions.MultivariateNormal(observation_mu, observation_cov)
        generative_likelihood = torch.sum(observation_normal_dist.log_prob(self.observations_seq))
        return generative_likelihood
