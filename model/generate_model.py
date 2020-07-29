#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from model import vaeakf_v1, vaeakf_v2, akf

def generate_model(args):

    Model_Factory = {
       '1': vaeakf_v1.VAEAKF,
       '2': vaeakf_v2.VAEAKF,
       '3': akf.AKF,
    }
    model = Model_Factory[str(args.version)](
        input_size=args.input_size,
        state_size=args.state_size,
        observation_size=args.observation_size,
        net_type=args.net_type,
        k=args.k_size,
        num_layers=args.num_layers,
        L=args.L
    )
    return model