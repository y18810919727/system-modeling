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
from model.deepar import DeepAR
#from model import vaeakf_combinational_linears_random as vaeakf_combinational_linears


def generate_model(args):

    if args.model.type == 'vaecl':
        #model = vaeakf_cl.VAEAKFCombinedLinear(
        model = vaeakf_combinational_linears.VAEAKFCombinedLinear(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
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
            filtering=args.model.filtering,
            D=args.model.D
        )
    elif args.model.type == 'vrnn':
        model = VRNN(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            net_type=args.model.net_type,
            k=args.model.k_size,
            num_layers=args.model.num_layers,
            D=args.model.D
        )
    elif args.model.type == 'deepar':
        model = DeepAR(
            input_size=args.dataset.input_size,
            observations_size=args.dataset.observation_size,
            net_type=args.model.net_type,
            k=args.model.k_size,
            num_layers=args.model.num_layers
        )
    elif args.model.type == 'seq2seq':
        from model.attention_seq2seq import AttentionSeq2Seq
        model = AttentionSeq2Seq(input_size=args.dataset.input_size, observations_size=args.dataset.observation_size,
                                 state_size=args.model.state_size, max_length=args.model.max_length,
                                 label_length=args.model.label_length, num_layers=args.model.num_layers,
                                 dropout_p=args.model.dropout_p, train_pred_len=args.model.train_pred_len)

    elif args.model.type == 'informer':
        from model.informer.model import Informer
        model = Informer(
            args.dataset.input_size,
            args.dataset.input_size + args.dataset.observation_size,
            args.dataset.observation_size,
            args.history_length,
            args.model.label_len,
            args.model.train_pred_len,
            args.model.factor,
            args.model.d_model,
            args.model.n_heads,
            args.model.e_layers, # self.args.e_layers,
            args.model.d_layers,
            args.model.d_ff,
            args.model.dropout,
            args.model.attn,
            'gelu',
            True,
            True,
            args.model.mix
        ).float()

    else:
        raise NotImplementedError
    return model
