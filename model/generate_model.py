#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
#from model import vaeakf_combinational_linears
from model import vaeakf_combinational_linears as vaeakf_combinational_linears
from model.srnn import SRNN
from model.vrnn import VRNN
#from model import vaeakf_combinational_linears_random as vaeakf_combinational_linears


def generate_model(args):

    if args.model.type == 'vaecl':
        #model = vaeakf_cl.VAEAKFCombinedLinear(
        model = vaeakf_combinational_linears.VAEAKFCombinedLinear(
            input_size=args.dataset.input_size,
            state_size=args.model.k_size,
            observations_size=args.dataset.observation_size,
            k=args.model.posterior.k_size,
            num_layers=args.model.posterior.num_layers,
            L=args.model.L,
            R=args.model.R,
            D=args.model.D,
            num_linears=args.model.dynamic.num_linears
        )
    elif args.model.type == 'srnn':
        model = SRNN(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            net_type=args.model.net_type,
            k=args.model.k_size,
            num_layers=args.model.num_layers,
            filtering=args.model.filtering
        )
    elif args.model.type == 'vrnn':
        model = VRNN(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            net_type=args.model.net_type,
            k=args.model.k_size,
            num_layers=args.model.num_layers,
        )

    else:
        raise NotImplementedError
    return model
