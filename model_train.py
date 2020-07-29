#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import time
import shutil
import sys


import pandas
import pandas
from config import args
from dataset import FakeDataset
from torch.utils.data import DataLoader
import traceback


def set_random_seed(config):

    if config.random_seed is None:
        rand_seed = np.random.randint(0,100000)
    else:
        rand_seed = config.random_seed
    print('random seed = {}'.format(rand_seed))
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def test_net(model, data_loader):

    acc_loss = 0
    acc_items = 0
    acc_mse = 0
    for i, data in enumerate(data_loader):

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

        acc_loss += float(loss)*external_input.shape[1]
        acc_items += external_input.shape[1]
        acc_mse += float(torch.nn.functional.mse_loss(
            state, model.sample_state(max_prob=True)
        ))*external_input.shape[1]

    return acc_loss/acc_items, acc_mse/acc_items


def main(args, logging):
    set_random_seed(args)
    from model.generate_model import generate_model
    model = generate_model(args)
    if args.use_cuda:
        model = model.cuda()

    logging(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    train_df = pandas.read_csv('data/fake.csv')
    val_df = pandas.read_csv('data/fake_val.csv')

    train_dataset = FakeDataset(train_df)
    val_dataset = FakeDataset(train_df)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    best_loss = 1e10

    logging('make train loader successfully')
    for epoch in range(args.epochs):
        acc_loss = 0
        acc_items = 0
        for i, data in enumerate(train_loader):
            t1 = time.time()

            external_input, observation, state,initial_mu, initial_sigma =data
            acc_items += external_input.shape[0]

            external_input = external_input.permute(1, 0, 2)
            observation = observation.permute(1, 0, 2)
            state = state.permute(1, 0, 2)

            if args.use_cuda:
                external_input = external_input.cuda()
                observation = observation.cuda()
                state = state.cuda()
                initial_mu = initial_mu.cuda()
                initial_sigma = initial_sigma.cuda()

            t2 = time.time()
            model(external_input, observation, initial_mu, initial_sigma)
            t3 = time.time()
            loss = model.call_loss()
            acc_loss += float(loss)*external_input.shape[1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging('epoch-round = {}-{} with loss = {:.4f} prepare time {:.4f} forward time {:.4f}, forward percent{:.4f}%'.format(
                epoch, i, float(loss), t2-t1, t3-t2, 100*(t3-t2)/(t3-t1)
            ))

        logging('epoch = {} with train_loss = {:.4f}'.format(
            epoch, float(acc_loss/acc_items)
        ))
        if (epoch + 1) % args.eval_epochs == 0:
            val_loss, val_mse = test_net(model, val_loader)
            logging('eval epoch = {} with loss = {:.4f} mse = {:.4f}'.format(
                epoch, val_loss, val_mse)
            )
            if best_loss > val_loss:
                best_loss = val_loss
                ckpt = dict()
                ckpt['model'] = model.state_dict()
                ckpt['epoch'] = epoch + 1
                torch.save(ckpt, os.path.join('ckpt', args.save_dir, 'best.pth'))
                logging('Save ckpt at epoch = {}'.format(epoch))




if __name__ == '__main__':

    from common import SimpleLogger
    save_dir_path = os.path.join('ckpt', args.save_dir)
    if os.path.exists(save_dir_path):
        if args.reset:
            shutil.rmtree(save_dir_path, ignore_errors=True)
        else:
            print('Exit; already completed and no hard reset asked.')
            sys.exit()  # do not overwrite folder with current experiment

    logging = SimpleLogger(os.path.join('ckpt', args.save_dir, 'log.out'))
    try:
        main(args, logging)
    except Exception as e:
        var = traceback.format_exc()
        logging(var)




