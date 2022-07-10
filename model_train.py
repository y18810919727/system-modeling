#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
import traceback


import torch
import time
import shutil
import sys

import pandas
import pandas as pd

from dataset import WesternDataset, WesternConcentrationDataset, CstrDataset, WindingDataset, IBDataset, WesternDataset_1_4, CTSample, NLDataset
from dataset import ActuatorDataset, BallbeamDataset, DriveDataset, DryerDataset, GasFurnaceDataset, SarcosArmDataset
from torch.utils.data import DataLoader
from lib import util
import hydra
from omegaconf import DictConfig, OmegaConf
import traceback
from scipy.stats import pearsonr
import common
from common import detect_download, init_network_weights, vae_loss, split_first_dim
from lib.util import TimeRecorder
from model.common import PeriodSchedule, ValStepSchedule
from numpy import *



# os.environ["CUDA_VISIBLE_DEVICES"] = str(3)


def set_random_seed(seed, logging):
    rand_seed = np.random.randint(0, 100000) if seed is None else seed
    logging('random seed = {}'.format(rand_seed))
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


def test_net(model, data_loader, epoch, args):
    acc_loss = 0
    acc_items = 0
    acc_rrse = 0
    acc_rmse = 0
    acc_time = 0
    acc_pred_likelihood = 0
    acc_multisample_rmse = 0

    model.eval()
    use_cuda = args.use_cuda and torch.cuda.is_available()

    tr = TimeRecorder()

    for i, data in enumerate(data_loader):

        external_input, observation = data
        external_input = external_input.permute(1, 0, 2)
        observation = observation.permute(1, 0, 2)
        if use_cuda:
            external_input = external_input.cuda()
            observation = observation.cuda()

        l, batch_size, _ = external_input.size()

        with tr('val'):
        # Update: 20210618 ，删掉训练阶段在model_train中调用forward_posterior的过程,直接调用call_loss(external_input, observation)
            losses = model.call_loss(external_input, observation)
            loss, kl_loss, likelihood_loss = losses['loss'], losses['kl_loss'], losses['likelihood_loss']

            if kl_loss != 0:
                loss = vae_loss(kl_loss, likelihood_loss, epoch, kl_inc=args.train.kl_inc,
                                kl_wait=args.train.kl_wait, kl_max=args.train.kl_max)

            # region Prediction
            prefix_length = max(int(args.dataset.history_length * args.sp),
                                1) if args.ct_time else args.dataset.history_length
            _, memory_state = model.forward_posterior(
                external_input[:prefix_length], observation[:prefix_length]
            )
            outputs, memory_state = model.forward_prediction(
                external_input[prefix_length:], n_traj=args.test.n_traj, memory_state=memory_state
            )

        acc_time += tr['val']

        acc_loss += float(loss) * external_input.shape[1]
        # 预测的likelihood
        pred_likelihood = - float(torch.sum(outputs['predicted_dist'].log_prob(observation[prefix_length:]))) / batch_size / (l - prefix_length)
        acc_pred_likelihood += pred_likelihood * external_input.shape[1]
        acc_items += external_input.shape[1]

        acc_rrse += float(common.RRSE(
            observation[prefix_length:], outputs['predicted_seq'])
        ) * external_input.shape[1]

        acc_rmse += float(common.RMSE(
            observation[prefix_length:], outputs['predicted_seq'])
        ) * external_input.shape[1]

        acc_multisample_rmse += mean([float(common.RMSE(
            observation[prefix_length:], outputs['predicted_seq_sample'][:, :, i, :])) for i in range(outputs['predicted_seq_sample'].shape[2])]
        ) * external_input.shape[1]

    model.train()
    return acc_loss / acc_items, acc_rrse / acc_items, acc_rmse/acc_items, acc_time / acc_items, \
           acc_pred_likelihood / acc_items, acc_multisample_rmse / acc_items


def main_train(args, logging):
    # 设置随机种子，便于结果复现
    global scale
    set_random_seed(args.random_seed, logging)
    use_cuda = args.use_cuda and torch.cuda.is_available()

    # 根据args的配置生成模型
    from model.generate_model import generate_model
    model = generate_model(args)

    # 模型加载到gpu上
    if use_cuda:
        model = model.cuda()

    init_network_weights(model)
    logging('save dir = {}'.format(os.getcwd()))
    logging(model)

    # 设置模型训练优化器
    if args.train.optim.type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.train.optim.lr)
    elif args.train.optim.type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.train.optim.lr)

    # 学习率调整器
    if args.train.schedule.type == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.train.schedule.gamma)
    elif args.train.schedule.type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.train.schedule.step_size, gamma=args.train.schedule.gamma)
    elif args.train.schedule.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train.schedule.T_max,
                                                               eta_min=args.train.schedule.eta_min)
    elif args.train.schedule.type == 'val_step':
        scheduler = ValStepSchedule(optimizer,
                                    args.train.schedule.lr_scheduler_nstart,
                                    args.train.schedule.lr_scheduler_nepochs,
                                    args.train.schedule.lr_scheduler_factor,
                                    logging)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1)
        logging('No scheduler used in training !!!!')

    if hasattr(args.train.schedule, 'step_size'):
        scheduler = PeriodSchedule(scheduler, args.train.schedule.step_size, logging)

    # 构建训练集和验证集

    access_key = pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), 'data', 'AccessKey.csv'))
    if args.dataset.type == 'fake':
        train_df = pandas.read_csv('data/fake_train.csv')
        val_df = pandas.read_csv('data/fake_val.csv')
        train_dataset = FakeDataset(train_df)
        val_dataset = FakeDataset(val_df)
    elif args.dataset.type.startswith('west'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data', 'west', 'data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/west')
        if not os.path.exists(base):
            os.mkdir(base)
        # 检测数据集路径，如果本地没有数据自动下载
        data_paths = detect_download(objects,
                                     base,
                                     'http://oss-cn-beijing.aliyuncs.com',
                                     'west-part-pressure',
                                     access_key['AccessKey ID'][0],
                                     access_key['AccessKey Secret'][0]
                                     )
        data_csvs = [pd.read_csv(path) for path in data_paths]
        dataset_split = [0.6, 0.2, 0.2]
        # 训练测试集的比例划分
        train_size, val_size, test_size = [int(len(data_csvs) * ratio) for ratio in dataset_split]
        if args.dataset.type.endswith('1_4'):
            train_dataset = WesternDataset_1_4(data_csvs[:train_size],
                                           args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window,
                                           dilation=args.dataset.dilation)
            val_dataset = WesternDataset_1_4(data_csvs[train_size:train_size + val_size],
                                         args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window,
                                         dilation=args.dataset.dilation)
        else:
            train_dataset = WesternDataset(data_csvs[:train_size],
                                           args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window,
                                           dilation=args.dataset.dilation)
            val_dataset = WesternDataset(data_csvs[train_size:train_size + val_size],
                                         args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window,
                                         dilation=args.dataset.dilation)
    elif args.dataset.type == 'west_con':
        data_dir = os.path.join(hydra.utils.get_original_cwd(), 'data/west_con')
        data_csvs = [pd.read_csv(os.path.join(data_dir, file)) for file in os.listdir(data_dir)]
        dataset_split = [0.6, 0.2, 0.2]
        train_size, val_size, test_size = [int(len(data_csvs) * ratio) for ratio in dataset_split]
        train_dataset = WesternConcentrationDataset(data_csvs[:train_size],
                                                    args.dataset.history_length + args.dataset.forward_length,
                                                    step=args.dataset.dataset_window,
                                                    dilation=args.dataset.dilation)
        val_dataset = WesternConcentrationDataset(data_csvs[train_size:train_size + val_size],
                                                  args.dataset.history_length + args.dataset.forward_length,
                                                  step=args.dataset.dataset_window,
                                                  dilation=args.dataset.dilation)
    elif args.dataset.type.startswith('cstr'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/cstr/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/cstr')
        if not os.path.exists(base):
            os.mkdir(base)
        # _ = detect_download(objects, base)
        _ = detect_download(objects,
                            base,
                            'http://oss-cn-beijing.aliyuncs.com',
                            'io-system-data',
                            access_key['AccessKey ID'][0],
                            access_key['AccessKey Secret'][0]
                            )
        train_dataset = CstrDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = CstrDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('actuator'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/actuator/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/actuator')
        if not os.path.exists(base):
            os.mkdir(base)

        train_dataset = ActuatorDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = ActuatorDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('ballbeam'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/ballbeam/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/ballbeam')
        if not os.path.exists(base):
            os.mkdir(base)

        train_dataset = BallbeamDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = BallbeamDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('drive'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/drive/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/drive')
        if not os.path.exists(base):
            os.mkdir(base)

        train_dataset = DriveDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = DriveDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('dryer'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/dryer/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/dryer')
        if not os.path.exists(base):
            os.mkdir(base)

        train_dataset = DryerDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = DryerDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('gas_furnace'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/gas_furnace/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/gas_furnace')
        if not os.path.exists(base):
            os.mkdir(base)

        train_dataset = GasFurnaceDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = GasFurnaceDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('sarcos'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/sarcos/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/sarcos')
        if not os.path.exists(base):
            os.mkdir(base)

        train_dataset = SarcosArmDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = SarcosArmDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('nl'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/nl/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/nl')
        if not os.path.exists(base):
            os.mkdir(base)
        train_dataset = NLDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = NLDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('ib'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/ib/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/ib')
        if not os.path.exists(base):
            os.mkdir(base)
        train_dataset = IBDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = IBDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        scale = train_dataset.normalize_record()

    elif args.dataset.type.startswith('winding'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/winding/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/winding')
        if not os.path.exists(base):
            os.mkdir(base)
        _ = detect_download(objects,
                            base,
                            'http://oss-cn-beijing.aliyuncs.com',
                            'io-system-data',
                            access_key['AccessKey ID'][0],
                            access_key['AccessKey Secret'][0]
                            )
        train_dataset = WindingDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = WindingDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('southeast'):
        from dataset import SoutheastThickener

        data = np.load(os.path.join(hydra.utils.get_original_cwd(), args.dataset.data_path))
        train_dataset = SoutheastThickener(data,
                                           length=args.dataset.history_length + args.dataset.forward_length,
                                           step=args.dataset.step,
                                           dataset_type='train', io=args.dataset.io,
                                           smooth_alpha=args.dataset.smooth_alpha
                                           )

        val_dataset = SoutheastThickener(data,
                                         length=args.dataset.history_length + args.dataset.forward_length,
                                         step=args.dataset.step,
                                         dataset_type='val', io=args.dataset.io, smooth_alpha=args.dataset.smooth_alpha
                                         )

    elif args.dataset.type.startswith('southeast'):

        from southeast_ore_dataset import SoutheastOreDataset

        dataset_split = [0.6, 0.2, 0.2]
        train_dataset, val_dataset, _, scaler = SoutheastOreDataset(
            data_dir=hydra.utils.get_original_cwd(),
            step_time=[args.dataset.in_length, args.dataset.out_length, args.dataset.window_step],
            data_from_csv=args.dataset.data_from_csv,
            in_name=args.dataset.in_columns,
            out_name=args.dataset.out_columns,
            logging=logging,
            ctrl_solution=args.ctrl_solution,
        ).get_split_dataset(dataset_split)

    else:
        raise NotImplementedError

    collate_fn = None if not args.ct_time else CTSample(args.sp, args.base_tp, evenly=args.sp_even).batch_collate_fn

    # 构建dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.train.batch_size,
                              shuffle=True, num_workers=args.train.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.train.batch_size, shuffle=False,
                            num_workers=args.train.num_workers, collate_fn=collate_fn)
    # best_rrse = 1e12
    # best_rmse = 1e12
    best_val = 1e12

    logging('make train loader successfully. Length of loader: %i' % len(train_loader))

    best_dev_epoch = -1

    # all_val_rrse = []
    all_val_rmse = []

    # 开始训练，重复执行args.train.epochs次
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

            if use_cuda:
                external_input = external_input.cuda()
                observation = observation.cuda()

            t2 = time.time()

            losses = model.call_loss(external_input, observation)
            loss, kl_loss, likelihood_loss = losses['loss'], losses['kl_loss'], losses['likelihood_loss']
            if kl_loss != 0:
                loss = vae_loss(kl_loss, likelihood_loss, epoch, kl_inc=args.train.kl_inc,
                                kl_wait=args.train.kl_wait, kl_max=args.train.kl_max)

            t3 = time.time()

            acc_loss += float(loss) * external_input.shape[1]
            acc_kl_loss += float(kl_loss) * external_input.shape[1]
            acc_likelihood_loss += float(likelihood_loss) * external_input.shape[1]

            optimizer.zero_grad()  # 清理梯度
            loss.backward()  # loss反向传播
            optimizer.step()  # 参数优化
            logging(
                'epoch-round = {}-{} with loss = {:.4f} kl_loss = {:.4f}  likelihood_loss = {:.4f} '
                'prepare time {:.4f} forward time {:.4f}, forward percent{:.4f}%'.format(
                    epoch, i, float(loss), float(kl_loss), float(likelihood_loss), t2 - t1, t3 - t2,
                                                                                   100 * (t3 - t2) / (t3 - t1))
            )
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logging('epoch = {} with train_loss = {:.4f} with kl_loss = {:.4f} with likelihood_loss = {:.4f} learning_rate = {:.6f}'.format(
            epoch, float(acc_loss / acc_items), float(acc_kl_loss / acc_items), float(acc_likelihood_loss / acc_items), lr
        ))
        if (epoch + 1) % args.train.eval_epochs == 0:
            with torch.no_grad():
                val_loss, val_rrse, val_rmse, val_time, val_pred_likelihood, val_multisample_rmse = test_net(model, val_loader, epoch, args)
            logging('eval epoch = {} with loss = {:.6f} rmse = {:.4f} rrse = {:.4f} val_time = {:.4f} val_pred_likelihood = {:.4f} val_multisample_rmse = {:.4f} learning_rate = {:.6f}'.format(
                epoch, val_loss, val_rmse, val_rrse, val_time, val_pred_likelihood, val_multisample_rmse, lr)
            )
            all_val_rmse.append(val_rmse)
            scheduler.step(all_val_rmse, val_rmse)
            # TODO:目前评价标准为rmse，需要同时考虑rmse\likelihood\multisample_rmse? 如何比较好的同时以三者为评价标准？ # 用likehood好一些，越大越好
            epoch_val = val_pred_likelihood
            if best_val > epoch_val:
                best_val = epoch_val
                best_dev_epoch = epoch
                ckpt = dict()
                ckpt['model'] = model.state_dict()
                ckpt['epoch'] = epoch + 1
                #  ckpt['scale'] = scale    # 记录训练数据的均值和方差用于控制部分归一化和反归一化
                torch.save(ckpt, os.path.join('./', 'best.pth'))
                torch.save(model.to(torch.device('cpu')), os.path.join('./', 'control.pkl'))
                if use_cuda:
                    model = model.cuda()
                logging('Save ckpt at epoch = {}'.format(epoch))

            if epoch - best_dev_epoch > args.train.max_epochs_stop and epoch > args.train.min_epochs:
                logging('Early stopping at epoch = {}'.format(epoch))
                break


        # Update learning rate
        if not args.train.schedule.type == 'val_step':
            scheduler.step()  # 更新学习率

        # lr - Early stoping condition
        if lr < args.train.optim.min_lr:
            logging('lr is too low! Early stopping at epoch = {}'.format(epoch))
            break

    logging('Training finished')


@hydra.main(config_path='config', config_name="config.yaml")
def main_app(args: DictConfig) -> None:
    from common import SimpleLogger, training_loss_visualization

    logging = SimpleLogger('./log.out')

    # region loading the specific model configuration (config/paras/{dataset}/{model}.yaml)
    if args.use_model_dataset_config:
        model_dataset_config = util.load_DictConfig(
            os.path.join(hydra.utils.get_original_cwd(), 'config', 'paras', args.dataset.type),
            args.model.type + '.yaml'
        )
        if model_dataset_config is None:
            logging(f'Can not find model config file  in config/paras/{args.dataset.type}/{args.model.type}.yaml, '
                    f'loading default model config')
        else:
            args.model = model_dataset_config
    # endregion

    # In continuous-time mode, the last dimension of input variable is the delta of time step.
    if args.ct_time:
        args.dataset.input_size += 1

    # Save args for running model_test.py individually
    util.write_DictConfig('./', 'exp.yaml', args)

    logging(OmegaConf.to_yaml(args))

    # Model Training
    try:
        main_train(args, logging)
        training_loss_visualization('./')
    except Exception as e:
        var = traceback.format_exc()
        logging(var)

    # Evaluation in Test Dataset
    from model_test import main_test
    ckpt_path = './'
    logging = SimpleLogger(
        os.path.join(
            ckpt_path, 'test.out'
        )
    )
    try:
        with torch.no_grad():
            main_test(args, logging, ckpt_path)
    except Exception as e:
        var = traceback.format_exc()
        logging(var)


if __name__ == '__main__':
    main_app()
