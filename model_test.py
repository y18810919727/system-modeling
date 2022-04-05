#!/usr/bin/python
# -*- coding:utf8 -*-
import math
import os
import json

import torch
import time

import pandas as pd
import numpy as np
from lib import util
from dataset import FakeDataset
from torch.utils.data import DataLoader
from dataset import WesternDataset, WesternConcentrationDataset, CstrDataset, WindingDataset, IBDataset, WesternDataset_1_4, CTSample, NLDataset
import traceback
from matplotlib import pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from common import normal_interval, Statistc
from southeast_ore_dataset import SoutheastOreDataset


def set_random_seed(seed):
    rand_seed = np.random.randint(0, 100000) if seed is None else seed
    print('random seed = {}'.format(rand_seed))
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


def main_test(args, logging, ckpt_path):
    set_random_seed(args.random_seed)
    from model.generate_model import generate_model
    figs_path = os.path.join(logging.dir, 'figs')
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    if args.test.plt_single:   # 单图绘制
        single_figs_path = os.path.join(figs_path, 'single_figs')
        if not os.path.exists(single_figs_path):
            os.makedirs(single_figs_path)

    model = generate_model(args)
    ckpt = torch.load(
        os.path.join(
            ckpt_path, 'best.pth'
        )
    )
    model.load_state_dict(ckpt['model'])
    if args.use_cuda:
        model = model.cuda()
    model.eval()
    logging(model)

    if args.dataset.type == 'fake':
        test_df = pd.read_csv(hydra.utils.get_original_cwd(), 'data/fake_test.csv')
        dataset = FakeDataset(test_df)

    elif args.dataset.type.startswith('west'):

        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data', 'west', 'data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/west')
        if not os.path.exists(base):
            os.mkdir(base)
        # 检测数据集路径，如果本地没有数据自动下载

        access_key = pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), 'data', 'AccessKey.csv'))
        from common import detect_download
        data_paths = detect_download(objects,
                                     base,
                                     'http://oss-cn-beijing.aliyuncs.com',
                                     'west-part-pressure',
                                     access_key['AccessKey ID'][0],
                                     access_key['AccessKey Secret'][0]
                                     )
        # data_csvs = [pd.read_csv(path) for path in data_paths]
        #
        # data_urls = pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), 'data/data_url.csv'))
        # base = os.path.join(hydra.utils.get_original_cwd(), 'data/part')
        # if not os.path.exists(base):
        #     os.mkdir(base)
        # from common import detect_download
        # data_paths = detect_download(data_urls, base)
        data_csvs = [pd.read_csv(path) for path in data_paths]
        dataset_split = [0.6, 0.2, 0.2]
        train_size, val_size, _ = [int(len(data_csvs) * ratio) for ratio in dataset_split]
        test_size = len(data_csvs) - train_size - val_size
        if args.dataset.type.endswith('1_4'):
            dataset = WesternDataset_1_4(data_csvs[-test_size:], args.dataset.history_length + args.dataset.forward_length,
                                     args.dataset.dataset_window, dilation=args.dataset.dilation)
        else:
            dataset = WesternDataset(data_csvs[-test_size:], args.dataset.history_length + args.dataset.forward_length,
                                     args.dataset.dataset_window, dilation=args.dataset.dilation)

    elif args.dataset.type == 'west_con':
        data_dir = os.path.join(hydra.utils.get_original_cwd(), 'data/west_con')
        data_csvs = [pd.read_csv(os.path.join(data_dir, file)) for file in os.listdir(data_dir)]

        dataset_split = [0.6, 0.2, 0.2]
        train_size, val_size, _ = [int(len(data_csvs) * ratio) for ratio in dataset_split]
        test_size = len(data_csvs) - train_size - val_size
        dataset = WesternConcentrationDataset(data_csvs[-test_size:], args.dataset.history_length + args.dataset.forward_length,
                                              args.dataset.dataset_window, dilation=args.dataset.dilation)
    elif args.dataset.type == 'cstr':
        dataset = CstrDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.test_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type == 'nl':
        dataset = NLDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.test_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type == 'ib':
        dataset = IBDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.test_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type == 'winding':
        dataset = WindingDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.test_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
    elif args.dataset.type == 'southeast':
        dataset_split = [0.6, 0.2, 0.2]
        _, _, dataset, scaler = SoutheastOreDataset(
            data_dir=hydra.utils.get_original_cwd(),
            step_time=[args.dataset.in_length, args.dataset.out_length, args.dataset.window_step],
            in_name=args.dataset.in_columns,
            out_name=args.dataset.out_columns,
            logging=logging,
            ctrl_solution=args.ctrl_solution,
            data_from_csv=True,
        ).get_split_dataset(dataset_split)

    else:
        raise NotImplementedError

    test_loader = DataLoader(dataset, batch_size=args.test.batch_size, shuffle=False,
                             num_workers=args.train.num_workers, drop_last=True,
                             collate_fn=CTSample(args.sp, args.base_tp).batch_collate_fn if args.ct_time else None)

    logging('make test loader successfully. Length of loader: %i' % len(test_loader))
    acc_loss = 0
    acc_rrse = 0
    acc_time = 0
    acc_name = ['likelihood', 'ob_rrse'] + \
               ['ob_{}_rrse'.format(name) for name in args.dataset.target_names] + \
               ['ob_{}_pear'.format(name) for name in args.dataset.target_names] + \
               ['pred_rrse'] + \
               ['pred_{}_rrse'.format(name) for name in args.dataset.target_names] + \
               ['pred_{}_pear'.format(name) for name in args.dataset.target_names] + \
               ['pred_rmse'] + \
               ['pred_{}_rmse'.format(name) for name in args.dataset.target_names] + \
               ['time']
    acc_info = np.zeros(len(acc_name))

    def single_data_generator(acc_info):
        for i, data in enumerate(test_loader):

            external_input, observation = data
            if args.dataset.type == 'southeast':
                inverse_ex_input = scaler.inverse_transform_input(external_input)
                inverse_out = scaler.inverse_transform_output(observation)

            external_input = external_input.permute(1, 0, 2)
            observation = observation.permute(1, 0, 2)

            if args.use_cuda:
                external_input = external_input.cuda()
                observation = observation.cuda()

            outputs, memory_state = model.forward_posterior(external_input, observation)

            decode_observations_dist = model.decode_observation(outputs, mode='dist')
            decode_observations = model.decode_observation(outputs, mode='sample')
            decode_observation_low, decode_observation_high = normal_interval(decode_observations_dist, 2)

            # region Prediction
            prefix_length = max(int(args.dataset.history_length * args.sp), 1) if args.ct_time else args.dataset.history_length
            _, memory_state = model.forward_posterior(
                external_input[:prefix_length], observation[:prefix_length]
            )
            outputs, memory_state = model.forward_prediction(
                external_input[prefix_length:], args.test.n_traj, memory_state=memory_state
            )
            pred_observations_dist = outputs['predicted_dist']
            pred_observations_sample = outputs['predicted_seq']
            pred_observations_sample_traj = outputs['predicted_seq_sample']
            if args.model.type == 'vaecl':
                weight_map = memory_state['weight_map']
            else:
                weight_map = torch.zeros((2, 1, 2))  # minimal shape
            # endregion

            pred_observation_low, pred_observation_high = normal_interval(pred_observations_dist, 2)

            # region statistic
            variable_data_list = [
                external_input,
                observation,
                pred_observations_sample,
                decode_observations,
                prefix_length,
                model
            ]

            loss, ob_rrse, ob_rrse_single, ob_pear, prediction_rrse, prediction_rrse_single,\
            prediction_pearsonr, prediction_rmse, prediction_rmse_single, time = Statistic(variable_data_list, split=False)

            ob_pearson_info = ' '.join(
                ['ob_{}_pear={:.4f}'.format(name, pear) for pear, name in zip(ob_pear, args.dataset.target_names)])
            pred_pearson_info = ' '.join(['pred_{}_pear={:.4f}'.format(name, pear) for pear, name in zip(
                prediction_pearsonr, args.dataset.target_names)])

            ob_rrse_info = ' '.join(
                ['ob_{}_rrse={:.4f}'.format(name, rrse) for rrse, name in
                 zip(ob_rrse_single, args.dataset.target_names)])
            pred_rrse_info = ' '.join(['pred_{}_rrse={:.4f}'.format(name, rrse) for rrse, name in zip(
                prediction_rrse_single, args.dataset.target_names)])
            pred_rmse_info = ' '.join(['pred_{}_rmse={:.4f}'.format(name, rmse) for rmse, name in zip(
                prediction_rmse_single, args.dataset.target_names)])

            log_str = 'seq = {} loss = {:.4f} ob_rrse={:.4f} ' + ob_rrse_info + ' ' + ob_pearson_info + \
                      ' pred_rrse={:.4f} ' + pred_rrse_info + ' ' + pred_pearson_info + \
                      ' pred_rmse={:.4f} ' + pred_rmse_info + ' time={:.4f}'
            logging(log_str.format(i, loss,
                                   ob_rrse,
                                   prediction_rrse,
                                   prediction_rmse,
                                   time))

            acc_info += np.array([
                loss, ob_rrse, *ob_rrse_single, *ob_pear,
                prediction_rrse, *prediction_rrse_single, *prediction_pearsonr,
                prediction_rmse, *prediction_rmse_single, time
            ], dtype=np.float32)

            for i in range(external_input.size()[1]):
                yield tuple([x[:, i:i + 1, :] for x in [observation, decode_observations,
                                                        decode_observation_low, decode_observation_high,
                                                        pred_observation_low, pred_observation_high,
                                                        pred_observations_sample, pred_observations_sample_traj, external_input]] + [weight_map])

    for i, result in enumerate(single_data_generator(acc_info)):
        if i % int(len(dataset) // args.test.plt_cnt) == 0:

            observation, decode_observations, decode_observation_low, decode_observation_high, \
            pred_observation_low, pred_observation_high, pred_observations_sample, pred_observations_sample_traj, external_input = [x for x in
                                                                                     result[:-1]]
            # 计算单条数据的预测指标
            prefix_length = observation.size(0) - pred_observations_sample.size(0)
            variable_data_list = [
                external_input,
                observation,
                pred_observations_sample,
                decode_observations,
                prefix_length,
                model
            ]
            acc_info_single = Statistic(variable_data_list, split=True)


            # 遍历每一个被预测指标
            for _ in range(len(args.dataset.target_names)):
                observation, decode_observations, decode_observation_low, decode_observation_high, \
                pred_observation_low, pred_observation_high, pred_observations_sample, pred_observations_sample_traj = [x[..., _] for
                                                                                         x in
                                                                                         result[:-2]]
                external_input = result[-2]
                weight_map = result[-1]
                target_name = args.dataset.target_names[_]
                # region 开始画图
                plt.figure(figsize=(10, 8))
                ##################图一:隐变量区间展示###########################

                plt.subplot(221)
                text_list = ['{}={:.4f}'.format(name, value) for name, value in
                             zip(acc_name, acc_info_single)]
                for pos, text in zip(np.linspace(0, 1, len(text_list) + 1)[:-1], text_list):
                    plt.text(0.2, pos, text)
                # plt.plot(observation, label='observation')
                # plt.plot(estimate_state, label='estimate')
                # plt.fill_between(range(len(state)), interval_low, interval_high, facecolor='green', alpha=0.2,
                #                  label='hidden')
                # if args.dataset == 'fake':
                #     plt.plot(state, label='gt')
                #     #plt.plot(observation)

                # # southeast_ore_dateset适配
                # external_input = external_input.detach().cpu().squeeze()
                #
                # plt.plot(range(external_input.shape[0]), external_input, label=args.dataset.in_columns[0])
                # # plt.plot(range(external_input.shape[0]), external_input[:, 1])
                # plt.legend()

                x_all = range(len(observation))
                x_prefix = range(prefix_length)
                x_suffix = range(prefix_length, observation.size()[0])
                x_suffix_plus = range(prefix_length - 1, observation.size()[0])
                if args.ct_time:
                    x_all = torch.cumsum(external_input[x_all, 0, -1], dim=0).cpu().numpy() / args.base_tp
                    x_prefix, x_suffix, x_suffix_plus = [x_all[x_inds] for x_inds in [x_prefix, x_suffix, x_suffix_plus]]


                ##################图二:生成观测数据展示###########################
                plt.subplot(222)
                observation = observation.detach().cpu().squeeze(dim=1)
                estimate_observation_low = decode_observation_low.cpu().squeeze().detach()
                estimate_observation_high = decode_observation_high.cpu().squeeze().detach()
                plt.plot(x_all, observation, label=target_name, zorder=2)
                plt.fill_between(x_all, estimate_observation_low,
                                 estimate_observation_high,
                                 facecolor='green', alpha=0.2, label='95%', zorder=1)
                if args.ct_time:
                    plt.scatter(x_all, observation, s=1, c='black', zorder=3)
                plt.legend()

                ##################图三:预测效果###########################
                plt.subplot(223)

                # observation = scaler.inverse_transform_output(observation)
                # pred_observation_low = scaler.inverse_transform_output(pred_observation_low)
                # pred_observation_high = scaler.inverse_transform_output(pred_observation_high)
                # pred_observations_sample = scaler.inverse_transform_output(pred_observations_sample)
                plt.plot(x_prefix, observation[:prefix_length], label='history')
                plt.plot(x_suffix_plus, observation[prefix_length - 1:], label='real', zorder=4)
                pred_observations_sample_traj = pred_observations_sample_traj.permute(2, 0, 1)
                pred_observations_sample_traj = pred_observations_sample_traj.detach().squeeze(dim=-1).cpu().numpy()
                plt_sample_cnt = min(args.test.plt_n_traj, pred_observations_sample_traj.shape[0])
                for n in range(plt_sample_cnt):
                    plt.plot(x_suffix_plus,
                             np.concatenate([[float(observation[prefix_length - 1])],
                                             pred_observations_sample_traj[n]]),
                             label=('prediction_sample' if n == 0 else None), color='grey', linewidth=0.3, alpha=0.9, zorder=2)
                plt.plot(x_suffix_plus,
                         np.concatenate([[float(observation[prefix_length - 1])],
                                         pred_observations_sample.detach().squeeze().cpu().numpy()]),
                         label='prediction', color='red', alpha=0.7, zorder=3)
                plt.fill_between(x_suffix,
                                 pred_observation_low.detach().squeeze().cpu().numpy(),
                                 pred_observation_high.detach().squeeze().cpu().numpy(),
                                 facecolor='lightgreen', alpha=0.4, label='95%',  zorder=1)
                if args.ct_time:
                    plt.scatter(x_suffix, pred_observations_sample.detach().squeeze().cpu().numpy(), s=1, c='black', zorder=3)
                plt.ylabel(target_name)
                plt.legend()

                ##################图四:weight map 热力图###########################
                plt.subplot(224)
                weight = weight_map.mean(dim=1)  # 沿着batch维度求平均
                weight = weight.transpose(1, 0)
                weight = weight.detach().cpu().numpy()
                # cs = plt.contourf(weight, cmap=plt.cm.hot)
                cs = plt.contourf(weight)
                cs.changed()
                plt.colorbar()
                plt.xlabel('Steps')
                plt.ylabel('Weight of linears')

                # endregion 画图结束
                plt.savefig(
                    os.path.join(
                        figs_path, str(i) + '_' + str(_) + '.png'
                    )
                )
                plt.close()

                # plt_single
                if args.test.plt_single:
                    plt.figure(figsize=(10, 8))

                    plt.plot(x_prefix, observation[:prefix_length], label='history')
                    plt.plot(x_suffix_plus, observation[prefix_length - 1:], label='real', zorder=4)
                    for n in range(int(pred_observations_sample_traj.shape[0])):
                        if n == 0:
                            plt.plot(x_suffix_plus,
                                     np.concatenate([[float(observation[prefix_length - 1])],
                                                     pred_observations_sample_traj[n]]),
                                     label='prediction_sample', color='grey', linewidth=0.3, alpha=0.9, zorder=2)
                        else:
                            plt.plot(x_suffix_plus,
                                     np.concatenate([[float(observation[prefix_length - 1])],
                                                     pred_observations_sample_traj[n]]),
                                     color='grey', linewidth=0.3, alpha=0.9, zorder=2)
                    plt.plot(x_suffix_plus,
                             np.concatenate([[float(observation[prefix_length - 1])],
                                             pred_observations_sample.detach().squeeze().cpu().numpy()]),
                             label='prediction', color='red', alpha=0.7, zorder=3)
                    plt.fill_between(x_suffix,
                                     pred_observation_low.detach().squeeze().cpu().numpy(),
                                     pred_observation_high.detach().squeeze().cpu().numpy(),
                                     facecolor='lightgreen', alpha=0.4, label='95%', zorder=1)
                    if args.ct_time:
                        plt.scatter(x_suffix, pred_observations_sample.detach().squeeze().cpu().numpy(), s=1, c='r',
                                    zorder=3)
                    plt.ylabel(target_name)
                    plt.legend()

                    plt.savefig(
                        os.path.join(
                            single_figs_path, str(i) + '_' + str(_) + '.pdf'
                        ), dpi=600
                    )
                    plt.close()

    logging(' '.join(
        ['{}={:.4f}'.format(name, value / len(test_loader)) for name, value in zip(acc_name, acc_info)]
    ))

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

    if not hasattr(args.test, 'test_dir') or args.test.test_dir is None:
        raise AttributeError('It should specify save_dir attribute in test mode!')

    # ckpt_path = '/code/SE-VAE/ckpt/southeast/seq2seq/seq2seq_ctrl_solution=2/2021-06-12_23-06-08'
    # ckpt_path = '/code/SE-VAE/ckpt/southeast/tmp/vaecl_/2021-06-18_20-52-31'
    ckpt_path = args.test.test_dir

    logging = SimpleLogger(os.path.join(ckpt_path, 'test.out'))

    # region load the config of original model
    exp_config = util.load_DictConfig(
        ckpt_path,
        'exp.yaml'
    )
    if exp_config is not None:
        exp_config.test = args.test
        args = exp_config

    # endregion
    logging(OmegaConf.to_yaml(args))

    try:
        with torch.no_grad():
            main_test(args, logging, ckpt_path)
    except Exception as e:
        var = traceback.format_exc()
        logging(var)


if __name__ == '__main__':
    main_app()
