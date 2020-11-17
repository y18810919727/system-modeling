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
import pandas as pd
from dataset import FakeDataset, WesternDataset, CstrDataset, WindingDataset
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import traceback
from scipy.stats import pearsonr
import common
from common import detect_download


def set_random_seed(seed):

    rand_seed = np.random.randint(0,100000) if seed is None else seed
    print('random seed = {}'.format(rand_seed))
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


def test_net(model, data_loader, args):
    acc_loss = 0
    acc_items = 0
    acc_rrse = 0
    acc_time = 0
    for i, data in enumerate(data_loader):

        external_input, observation = data
        external_input = external_input.permute(1, 0, 2)
        observation = observation.permute(1, 0, 2)
        if args.use_cuda:
            external_input = external_input.cuda()
            observation = observation.cuda()

        begin_time = time.time()
        model.forward_posterior(external_input, observation)
        acc_time += time.time() - begin_time
        loss, _, _ = model.call_loss()

        acc_loss += float(loss)*external_input.shape[1]
        acc_items += external_input.shape[1]

        acc_rrse += float(common.RRSE(
            observation, model.decode(model.sample_state(max_prob=False), mode='sample'))
        )*external_input.shape[1]

    return acc_loss/acc_items, acc_rrse/acc_items, acc_time/acc_items


def main(args, logging):
    set_random_seed(args.random_seed)
    from model.generate_model import generate_model
    model = generate_model(args)
    if args.use_cuda:
        model = model.cuda()

    logging('save dir = {}'.format(os.getcwd()))
    logging(model)
    if args.train.optim.type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.train.optim.lr)
    elif args.train.optim.type == 'sgd':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.train.optim.lr)

    if args.train.schedule.type is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1)
        logging('No scheduler used in training !!!!')
    elif args.train.schedule.type == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.train.schedule.gamma)
    elif args.train.schedule.type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.train.schedule.step_size, gamma=args.train.schedule.gamma)
    elif args.train.schedule.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train.schedule.T_max, eta_min=args.train.schedule.eta_min)

    if args.dataset.type == 'fake':
        train_df = pandas.read_csv('data/fake_train.csv')
        val_df = pandas.read_csv('data/fake_val.csv')
        train_dataset = FakeDataset(train_df)
        val_dataset = FakeDataset(val_df)
    elif args.dataset.type == 'west':
        data_urls = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/part')
        if not os.path.exists(base):
            os.mkdir(base)
        data_paths = detect_download(data_urls, base)
        data_csvs = [pd.read_csv(path) for path in data_paths]
        dataset_split = [0.6, 0.2, 0.2]
        train_size, val_size, test_size = [int(len(data_csvs)*ratio) for ratio in dataset_split]
        train_dataset = WesternDataset(data_csvs[:train_size],
                                       args.history_length + args.forward_length, step=args.dataset.dataset_window)
        val_dataset = WesternDataset(data_csvs[train_size:train_size + val_size],
                                     args.history_length + args.forward_length, step=args.dataset.dataset_window)
    elif args.dataset.type.startswith('cstr'):
        data_urls = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/cstr/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/cstr')
        if not os.path.exists(base):
            os.mkdir(base)
        _ = detect_download(data_urls, base)
        train_dataset = CstrDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.history_length + args.forward_length, step=args.dataset.dataset_window)
        val_dataset = CstrDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.history_length + args.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('winding'):
        data_urls = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/winding/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/winding')
        if not os.path.exists(base):
            os.mkdir(base)
        _ = detect_download(data_urls, base)
        train_dataset = WindingDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.history_length + args.forward_length, step=args.dataset.dataset_window)
        val_dataset = WindingDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.history_length + args.forward_length, step=args.dataset.dataset_window)

    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size=args.train.batch_size,
                              shuffle=True, num_workers=args.train.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.train.batch_size, shuffle=False,
                            num_workers=args.train.num_workers)
    best_loss = 1e12

    logging('make train loader successfully')

    best_dev_epoch = -1
    for epoch in range(args.train.epochs):
        acc_loss = 0
        acc_kl_loss = 0
        acc_likelihood_loss = 0
        acc_items = 0
        for i, data in enumerate(train_loader):
            t1 = time.time()

            external_input, observation = data
            acc_items += external_input.shape[0]

            external_input = external_input.permute(1, 0, 2)
            observation = observation.permute(1, 0, 2)

            if args.use_cuda:
                external_input = external_input.cuda()
                observation = observation.cuda()

            t2 = time.time()
            model.forward_posterior(external_input, observation)
            t3 = time.time()

            loss, kl_loss, likelihood_loss = model.call_loss()
            acc_loss += float(loss) * external_input.shape[1]
            acc_kl_loss += float(kl_loss) * external_input.shape[1]
            acc_likelihood_loss += float(likelihood_loss) * external_input.shape[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging(
                'epoch-round = {}-{} with loss = {:.4f} kl_loss = {:.4f}  likelihood_loss = {:.4f} '
                'prepare time {:.4f} forward time {:.4f}, forward percent{:.4f}%'.format(
                    epoch, i, float(loss), float(kl_loss), float(likelihood_loss), t2-t1, t3-t2, 100*(t3-t2)/(t3-t1))
            )

        logging('epoch = {} with train_loss = {:.4f} with kl_loss = {:.4f} with likelihood_loss = {:.4f}'.format(
            epoch, float(acc_loss/acc_items), float(acc_kl_loss/acc_items), float(acc_likelihood_loss/acc_items)
        ))
        if (epoch + 1) % args.train.eval_epochs == 0:
            with torch.no_grad():
                val_loss, val_rrse, val_time = test_net(model, val_loader, args)
            logging('eval epoch = {} with loss = {:.4f} rrse = {:.4f} val_time = {:.4f}'.format(
                epoch, val_loss, val_rrse, val_time)
            )
            if best_loss > val_loss:
                best_loss = val_loss
                best_dev_epoch = epoch
                ckpt = dict()
                ckpt['model'] = model.state_dict()
                ckpt['epoch'] = epoch + 1
                torch.save(ckpt, os.path.join('./', 'best.pth'))
                logging('Save ckpt at epoch = {}'.format(epoch))

            if epoch - best_dev_epoch > args.train.max_epochs_stop:
                logging('Early stopping at epoch = {}'.format(epoch))
                break

        # Update learning rate
        scheduler.step()

    def is_parameters_printed(parameter):
        if 'estimate' in parameter[0]:
            return True
        return False

    logging(list(
        filter(
            is_parameters_printed,
            model.named_parameters())
    ))


@hydra.main(config_path='config', config_name="config.yaml")
def main_app(args: DictConfig) -> None:

    from common import SimpleLogger
    logging = SimpleLogger('./log.out')
    logging(OmegaConf.to_yaml(args))
    try:
        main(args, logging)
    except Exception as e:
        var = traceback.format_exc()
        logging(var)


if __name__ == '__main__':
    main_app()






