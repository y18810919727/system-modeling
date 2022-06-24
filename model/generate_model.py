#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
# from model import vaeakf_combinational_linears
from model import vaeakf_combinational_linears_random as vaeakf_combinational_linears
from model import VRNN, VAERNN, STORN, RNN, SRNN, ODERSSM, RSSM, TimeAwareRNN, DeepAR, STORN_SQRT, ODE_RNN, ODEVRNN
# from model.ct_model.ode_rssm_merge import ODERSSM_Merge


def generate_model(args):
    if args.model.type == 'vaecl':
        # model = vaeakf_cl.VAEAKFCombinedLinear(
        model = vaeakf_combinational_linears.VAEAKFCombinedLinear(
            input_size=args.dataset.input_size - (
                1 if args.dataset.type == 'southeast' and args.ctrl_solution == 2 else 0),
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size + (
                1 if args.dataset.type == 'southeast' and args.ctrl_solution == 2 else 0),
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
    elif args.model.type == 'rssm':
        model = RSSM(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            k=args.model.k_size,
            num_layers=args.model.num_layers,
            D=args.model.D
        )
    elif args.model.type == 'vaernn':
        model = VAERNN(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            k=args.model.k_size,
            num_layers=args.model.num_layers
        )
    elif args.model.type == 'rnn':
        model = RNN(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            k=args.model.k_size,
            num_layers=args.model.num_layers,
            ct_time=args.ct_time,
            sp=args.sp
        )
    elif args.model.type == 'storn':
        model = STORN(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            k=args.model.k_size,
            num_layers=args.model.num_layers,
        )
    elif args.model.type == 'storn_sqrt':
        model = STORN_SQRT(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            k=args.model.k_size,
            num_layers=args.model.num_layers,
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
            args.dataset.history_length,
            args.model.label_len,
            args.model.train_pred_len,
            args.model.factor,
            args.model.d_model,
            args.model.n_heads,
            args.model.e_layers,  # self.args.e_layers,
            args.model.d_layers,
            args.model.d_ff,
            args.model.dropout,
            args.model.attn,
            'gelu',
            True,
            True,
            args.model.mix
        ).float()
    elif args.model.type == 'ode_rssm':
        assert args.ct_time
        model = ODERSSM(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            ode_num_layers=args.model.ode_num_layers,
            k=args.model.k_size,
            D=args.model.D,
            ode_hidden_dim=args.model.ode_hidden_dim,
            ode_solver=args.model.ode_solver,
            rtol=args.model.rtol,
            atol=args.model.atol,
            ode_type=args.model.ode_type,
            weight=args.model.weight,
            detach=args.model.detach,
            ode_ratio=args.model.ode_ratio,
            iw_trajs=args.model.iw_trajs,
            odernn_encoder=args.model.odernn_encoder,
            z_in_ode=args.model.z_in_ode
        )
    elif args.model.type == 'ode_vrnn':
        assert args.ct_time
        model = ODEVRNN(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            ode_num_layers=args.model.ode_num_layers,
            k=args.model.k_size,
            D=args.model.D,
            ode_hidden_dim=args.model.ode_hidden_dim,
            ode_solver=args.model.ode_solver,
            rtol=args.model.rtol,
            atol=args.model.atol,
            ode_type=args.model.ode_type,
            weight=args.model.weight,
            detach=args.model.detach,
            ode_ratio=args.model.ode_ratio,
            iw_trajs=args.model.iw_trajs,
            ob_bptt=args.model.ob_bptt
        )
    elif args.model.type == 'time_aware':
        assert args.ct_time
        model = TimeAwareRNN(
            k_in=args.dataset.input_size,
            k_out=args.dataset.observation_size,
            k_state=args.model.state_size
        )
    elif args.model.type == 'ode_rnn':
        assert args.ct_time
        model = ODE_RNN(
            input_size=args.dataset.input_size,
            output_size=args.dataset.observation_size,
            hidden_size=args.model.k_size,
            ode_hidden_dim=args.model.ode_hidden_size,
            ode_solver=args.model.ode_solver,
            ode_num_layers=args.model.ode_num_layers,
        )

    # def __init__(self, k_in, k_out, k_state, dropout=0., cell_factory=GRUCell, interpol="constant", **kwargs):
    else:
        raise NotImplementedError
    return model
