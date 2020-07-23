#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import time


import pandas
import pandas
from config import args
from dataset import FakeDataset
from torch.utils.data import DataLoader
import traceback
from matplotlib import pyplot as plt


def set_random_seed(config):

    if config.random_seed is None:
        rand_seed = np.random.randint(0,100000)
    else:
        rand_seed = config.random_seed
    print('random seed = {}'.format(rand_seed))
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


def main(args, logging):
    set_random_seed(args)
    from model.generate_model import generate_model
    figs_path = os.path.join(logging.dir, 'figs')
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    model = generate_model(args)
    ckpt = torch.load(
        os.path.join(
            'ckpt', args.save_dir, str(args.test_id)+'.pth'
        )
    )
    model.load_state_dict(ckpt['model'])
    if args.use_cuda:
        model = model.cuda()
    logging(model)

    test_df = pandas.read_csv('data/fake2.csv')
    dataset = FakeDataset(test_df)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    logging('make test loader successfully')
    acc_loss = 0
    acc_mse = 0
    for i, data in enumerate(test_loader):

        external_input, observation, state,initial_mu, initial_sigma =data
        external_input = external_input.permute(1, 0, 2)
        observation = observation.permute(1, 0, 2)
        state = state.permute(1, 0, 2)

        if args.use_cuda:
            external_input = external_input.cuda()
            observation = observation.cuda()
            state = state.cuda()
            initial_mu = initial_mu.cuda()
            initial_sigma = initial_sigma.cuda()

        model(external_input, observation, initial_mu, initial_sigma)
        loss = model.call_loss()
        estimate_state = model.sample_state(max_prob=True)
        mse = torch.mean((estimate_state-state)**2)
        logging('seq = {} likelihood = {:.4f} mse={:.4f}'.format(
            i, float(loss), float(mse)
        ))
        acc_loss += float(loss)
        acc_mse += float(mse)
        if i<args.plt_cnt:
            observation = observation.detach().cpu().squeeze()
            state = state.detach().cpu().squeeze()
            estimate_state = estimate_state.detach().cpu().squeeze()

            assert len(state.shape) == 1
            plt.figure()
            plt.plot(state)
            plt.plot(estimate_state)
            plt.plot(observation)
            plt.legend(['real', 'estimate', 'observation'])
            plt.savefig(
                os.path.join(
                    figs_path, str(i)+'.png'
                )
            )


    logging('likelihood = {} mse = {:.4f}'.format(
       acc_loss/len(test_loader), float(acc_mse/len(test_loader))
    ))


if __name__ == '__main__':

    from common import SimpleLogger
    logging = SimpleLogger(os.path.join('ckpt', args.save_dir, 'test.out'))
    try:
        main(args, logging)
    except Exception as e:
        var = traceback.format_exc()
        logging(var)




