#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
from model.akf_vb import AKFVB


class AKF(AKFVB):
    """
    Mehra, R. K. (1972). Approaches to Adaptive Filtering. IEEE Transactions on Automatic Control, 17(5), 693–698. https://doi.org/10.1109/TAC.1972.1100100
    """
    def multivariateNormal_kl_loss(self, mu1, cov1, mu2, cov2):
        """
        In adaptive kalman filer, the posterior distribution of hidden states is solved theoretically.
        the variational bayes is unnecessary.
        The loss function becomes to the likelihood of observed data.
        More details can be found in Mehra, R. K. (1972). Approaches to Adaptive Filtering. IEEE Transactions on Automatic Control, 17(5), 693–698. https://doi.org/10.1109/TAC.1972.1100100
        """
        return torch.tensor(0).to(mu1.device)

