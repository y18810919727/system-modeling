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
from dataset import WesternDataset
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
            'ckpt', args.save_dir, 'best.pth'
        )
    )
    model.load_state_dict(ckpt['model'])
    if args.use_cuda:
        model = model.cuda()
    logging(model)


    if args.dataset == 'fake':
        test_df = pandas.read_csv('data/fake_test.csv')
        dataset = FakeDataset(test_df)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    elif args.dataset == 'west':

        test_datapath = 'res_2019.1003-1011.csv'
        from common import detect_and_download
        detect_and_download(test_datapath)
        test_df = pandas.read_csv(os.path.join('data', test_datapath))
        dataset = WesternDataset(test_df, args.length)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        raise NotImplementedError

    logging('make test loader successfully')
    acc_loss = 0
    acc_mse = 0
    acc_time = 0
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

        beg_time = time.time()
        model(external_input, observation, initial_mu, initial_sigma)
        end_time = time.time()
        acc_time += end_time - beg_time
        loss = model.call_loss()
        estimate_state = model.sample_state(max_prob=True)
        from common import cal_pearsonr
        mse = float(cal_pearsonr(estimate_state, state)[0])
        logging('seq = {} likelihood = {:.4f} mse={:.4f} time={:.4f}'.format(
            i, float(loss), float(mse), end_time - beg_time
        ))
        acc_loss += float(loss)
        acc_mse += float(mse)
        if i % int(len(test_loader)//args.plt_cnt) == 0:
            observation = observation.detach().cpu().squeeze()
            state = state.detach().cpu().squeeze()
            interval_low, interval_high = model.sigma_interval(1)
            interval_high = interval_high.cpu().squeeze().detach()
            interval_low = interval_low.cpu().squeeze().detach()
            estimate_state = estimate_state.detach().cpu().squeeze()

            assert len(state.shape) == 1

            plt.figure()
            plt.plot(observation, label='observation')
            plt.plot(estimate_state, label='estimate')
            plt.fill_between(range(len(state)), interval_low, interval_high, facecolor='green', alpha=0.2)
            if args.dataset == 'fake':
                plt.plot(state, label='gt')
                #plt.plot(observation)
            plt.legend()
            plt.savefig(
                os.path.join(
                    figs_path, str(i)+'.png'
                )
            )
            plt.close()


    logging('likelihood = {} mse = {:.4f} time = {:.4f}'.format(
       acc_loss/len(test_loader), float(acc_mse/len(test_loader)), acc_time/len(test_loader)
    ))
    logging('B =', str(model.estimate_B))
    logging('H =', str(model.estimate_H))
    logging('sigma =', str(torch.exp(model.estimate_logsigma)))
    logging('delta =', str(torch.exp(model.estimate_logdelta)))
    logging('bias =', str(torch.exp(model.estimate_bias)))


if __name__ == '__main__':

    from common import SimpleLogger
    logging = SimpleLogger(os.path.join('ckpt', args.save_dir, 'test.out'))
    try:
        main(args, logging)
    except Exception as e:
        var = traceback.format_exc()
        logging(var)




