#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import time
import json
import pandas as pd
import torch
import logging
from torch import nn
from numpy import *
from typing import Optional, Union
from sklearn.metrics import mean_squared_error


class SimpleLogger(object):
    def __init__(self, f, header='#logger output'):
        dir = os.path.dirname(f)
        self.dir = dir
        self.begin_time_sec = time.time()
        # print('test dir', dir, 'from', f)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(f, 'w') as fID:
            fID.write('%s\n' % header)
        self.f = f

    def __call__(self, *args):
        # standard output
        print(*args)
        # log to file
        try:
            with open(self.f, 'a') as fID:
                fID.write('Time_sec = {:.1f} '.format(time.time() - self.begin_time_sec))
                fID.write(' '.join(str(a) for a in args) + '\n')
        except:
            print('Warning: could not log to', self.f)


class TimeRecorder:
    def __init__(self):
        self.infos = {}

    def __call__(self, info, *args, **kwargs):
        class Context:
            def __init__(self, recoder, info):
                self.recoder = recoder
                self.begin_time = None
                self.info = info

            def __enter__(self):
                self.begin_time = time.time()

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.recoder.infos[self.info] = time.time() - self.begin_time

        return Context(self, info)

    def __str__(self):
        return ' '.join(['{}:{:.2f}s'.format(info, t) for info, t in self.infos.items()])

def normalize_seq(x, dim=0):
    import torch
    eps = 1e-12
    res = (x - torch.mean(x, dim=dim)) / torch.sqrt(torch.var(x, dim=dim)).clamp_min(eps)
    # assert torch.mean(torch.mean(res,dim=dim)).norm() <1e-4
    # assert (torch.mean(torch.var(res,dim=dim))-1).norm() <1e-4
    return res


def cal_pearsonr(tensor_seq1, tensor_seq2):
    """

    Args:
        tensor_seq1: shape (len, xxx)
        tensor_seq2: shape (len, xxx)

    Returns:

    """

    if len(tensor_seq1.shape) >= 2:
        pearson_list = [cal_pearsonr(tensor_seq1[:, i], tensor_seq2[:, i]) for i in range(tensor_seq1.shape[1])]
        return np.mean(pearson_list)
    from scipy.stats import pearsonr

    seq1 = tensor_seq1.detach().cpu().numpy()
    seq2 = tensor_seq2.detach().cpu().numpy()
    return pearsonr(seq1, seq2)[0]


def RRSE(y_gt, y_pred):
    assert y_gt.shape == y_pred.shape
    if len(y_gt.shape) == 3:
        return torch.mean(
            torch.stack(
                [RRSE(y_gt[:, i], y_pred[:, i]) for i in range(y_gt.shape[1])]
            )
        )

    elif len(y_gt.shape) == 2:
        # each shape (n_seq, n_outputs)
        se = torch.sum((y_gt - y_pred) ** 2, dim=0)
        rse = se / (torch.sum(
            (y_gt - torch.mean(y_gt, dim=0)) ** 2, dim=0
        )+1e-6)
        return torch.mean(torch.sqrt(rse))
    else:
        raise AttributeError


def RMSE(y_gt, y_pred):
    if len(y_gt.shape) == 3:
        return torch.mean(
            torch.stack(
                [RMSE(y_gt[:, i], y_pred[:, i]) for i in range(y_gt.shape[1])]
            )
        )
    elif len(y_gt.shape) == 2:
        se = torch.sum((y_gt - y_pred) ** 2, dim=0)
        rse = torch.sqrt(se / y_gt.shape[0])
        return torch.mean(rse)
    else:
        raise AttributeError


def Statistic(variable_data, split=False):
    """
    region statistic
    Args:
        variable_data:[
            external_input,
            observation,
            pred_observations_sample,
            decode_observations,
            prefix_length,  # history_length
            model
        ]
    Returns:

    """
    begin_time = time.time()
    observation, pred_observations_dist, pred_observations_sample, pred_observations_sample_traj, decode_observations, prefix_length = variable_data

    # 每条数据单独算loss太慢了

    # 统计参数1： 预测的rrse
    prediction_rrse = RRSE(observation[prefix_length:], pred_observations_sample)
    prediction_rrse_single = [float(RRSE(
        observation[prefix_length:, :, _], pred_observations_sample[:, :, _]
    )) for _ in range(observation.shape[-1])]
    # 统计参数2： 预测的pearson
    prediction_pearsonr = [float(cal_pearsonr(
        observation[prefix_length:, :, _], pred_observations_sample[:, :, _]
    )) for _ in range(observation.shape[-1])]
    # 统计参数3：重构RRSE
    ob_rrse = RRSE(
        observation, decode_observations
    )
    ob_rrse_single = [float(RRSE(
        observation[:, :, _], decode_observations[:, :, _]
    )) for _ in range(observation.shape[-1])]
    # 统计参数4：重构pearson
    ob_pear = [float(cal_pearsonr(
        observation[:, :, _], decode_observations[:, :, _]
    )) for _ in range(observation.shape[-1])]

    # 统计参数5: 真实序列在预测分布上的似然
    pred_likelihood = torch.sum(pred_observations_dist.log_prob(observation[prefix_length:]))
    pred_likelihood = - pred_likelihood / observation[prefix_length:].size()[0] / observation[prefix_length:].size()[1]

    # 统计参数6：预测的rmse
    prediction_rmse = RMSE(observation[prefix_length:], pred_observations_sample)
    prediction_rmse_single = [float(RMSE(
        observation[prefix_length:, :, _], pred_observations_sample[:, :, _]
    )) for _ in range(observation.shape[-1])]

    # 统计参数7：预测的multisample_rmse
    pred_multisample_rmse = mean([float(RMSE(
        observation[prefix_length:], pred_observations_sample_traj[:, :, i, :])) for i in range(pred_observations_sample_traj.shape[2])]
    )

    pred_multisample_rmse_single = [mean([float(RMSE(
        observation[prefix_length:, :, _], pred_observations_sample_traj[:, :, i, _])) for i in range(pred_observations_sample_traj.shape[2])]
    ) for _ in range(observation.shape[-1])]

    end_time = time.time()

    if split:
        return np.array([float(ob_rrse), *ob_rrse_single, *ob_pear, float(prediction_rrse), *prediction_rrse_single,
                         *prediction_pearsonr, float(pred_likelihood), float(prediction_rmse), *prediction_rmse_single,
                         float(pred_multisample_rmse), *pred_multisample_rmse_single, end_time - begin_time], dtype=np.float32)
    else:
        return [float(ob_rrse), ob_rrse_single, ob_pear, float(prediction_rrse), prediction_rrse_single,
                prediction_pearsonr, float(pred_likelihood), float(prediction_rmse), prediction_rmse_single,
                float(pred_multisample_rmse), pred_multisample_rmse_single, end_time - begin_time]

    # return [float(ob_rrse), ob_rrse_single, ob_pear,
    #         float(prediction_rrse), prediction_rrse_single, prediction_pearsonr, float(prediction_rmse), prediction_rmse_single, end_time - begin_time]


def softplus(x, threshold=20):
    return torch.where(
        x < threshold, torch.log(1 + torch.exp(x)), x
    )


def sqrt_softplus(x, threshold=20):
    return x.exp().sqrt()


def inverse_softplus(x, threshold=20):
    return torch.where(
        x < threshold, torch.log(torch.exp(x) - torch.ones_like(x)), x
    )


def logsigma2cov(logsigma):
    return torch.diag_embed(softplus(logsigma) ** 2)


def sqrt_logsigma2cov(logsigma):
    return torch.diag_embed(sqrt_softplus(logsigma))


def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger


def cov2logsigma(cov):
    return inverse_softplus(torch.sqrt(cov.diagonal(dim1=-2, dim2=-1)))


def normal_interval(dist, e):
    return dist.loc - e * torch.sqrt(dist.covariance_matrix.diagonal(dim1=-2, dim2=-1)), \
           dist.loc + e * torch.sqrt(dist.covariance_matrix.diagonal(dim1=-2, dim2=-1))


def cal_time(fn):
    """计算性能的修饰器"""

    def wrapper(*args, **kwargs):
        starTime = time.time()
        f = fn(*args, **kwargs)
        endTime = time.time()
        print('%s() runtime:%s ms' % (fn.__name__, 1000 * (endTime - starTime)))
        return f

    return wrapper


def detect_download(objects, base, oss_endpoint, bucket_name, accessKey_id, accessKey_secret):
    import oss2

    def download(bucket, name, path):
        """
        Data is stored in Aliyun OSS
        Args:
            bucket:
            name: name in oss
            path: path of the downloaded local file
        Returns: path if successful, else None
        """
        import urllib
        print("Downloading file %s:" % name)
        try:
            # urllib.request.urlretrieve(url, filename=os.path.join( path))
            bucket.get_object_to_file(name, path)
            return path
        except Exception as e:
            print("Error occurred when downloading file %s from %s, error message :" % (path, bucket))
            return None

    auth = oss2.Auth(accessKey_id, accessKey_secret)
    bucket = oss2.Bucket(auth, oss_endpoint, bucket_name)
    data_paths = []
    for name in objects['object']:
        path = os.path.join(base, name)
        if os.path.exists(path) or download(bucket, name, path):
            data_paths.append(path)

    return data_paths


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)


def merge_first_two_dims(tensor):
    size = tensor.size()
    return tensor.contiguous().reshape(-1, *size[2:])


def split_first_dim(tensor, sizes=None):
    if sizes is None:
        sizes = tensor.size()[0]
    import numpy
    assert type(sizes) is tuple and numpy.prod(sizes) == tensor.size()[0]
    return tensor.contiguous().reshape(*sizes, *tensor.size()[1:])


def read_std_mean(name, file_path, th_id):
    """
    get z-score statistical value to denormalize.
    Args:
        name: point name (eg. out_f, out_c, pressure)
        file_path: stat file path
        th_id: 1 or 2

    Returns: (mean, std)

    """
    stat_df = pd.read_csv(file_path, index_col=0)
    return stat_df.at['mean' + th_id, name], stat_df.at['std' + th_id, name]


def onceexp(data,a):#y一次指数平滑法
    x=[]
    t=data[0]
    x.append(t)
    for i in range(len(data)-1):
        t=t+a*(data[i+1]-t)
        x.append(t)
    return np.stack(x)


def subsample_indexes(c, time_steps, percentage, evenly=False):
    bs, l, d = c.shape
    n_to_subsample = int(l * percentage)
    if evenly:
        subsampled_idx = np.arange(l)[::int((l-1)/(n_to_subsample-1))] if n_to_subsample !=0 else np.arange(1)[0:1]
    else:
        subsampled_idx = sorted(np.random.choice(np.arange(l), n_to_subsample, replace=False))
    new_c = c[:,subsampled_idx]
    new_time_steps = time_steps[subsampled_idx]
    return new_c, new_time_steps


def vae_loss(kl_loss, rec_loss, epoch, kl_inc=False, kl_wait=10, kl_max=1.0):
    if not kl_inc:
        return kl_loss + rec_loss
    if epoch < kl_wait:
        kl_coef = 0.
    else:
        kl_coef = kl_max * (1-0.99 ** (epoch - kl_wait))
    return kl_coef * kl_loss + rec_loss


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val

class EMAMetric(object):
    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMetric, self).__init__()
        self._val = 0.
        self._gamma = gamma

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        self._val = self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    @property
    def val(self):
        return self._val

def training_loss_visualization(base_dir):
    print(base_dir)
    from matplotlib import pyplot as plt
    import re
    import os
    x_dataset_time = []
    y_dataset_train_loss = []
    y_dataset_kl_loss = []
    y_dataset_likelihood_loss = []
    x_dataset_eval_epoch = []
    y_dataset_eval_loss = []
    y_dataset_eval_train_loss = []
    train_epochs = []
    eval_rrse = []
    f = open(os.path.join(base_dir, 'log.out'))
    data = f.readlines()
    f.close()
    for line in data:
        try:
            if re.search('train_loss', line):
                pattern = re.compile(r'-?[0-9]\d*\.?\d*')  # 查找数字
                result = pattern.findall(line)
                x_dataset_time.append(float(result[0]))
                train_epochs.append(int(result[1]))
                y_dataset_train_loss.append(float(result[2]))
                y_dataset_kl_loss.append(float(result[3]))
                y_dataset_likelihood_loss.append(float(result[4]))
            elif re.search('eval', line) and line.startswith('Time_sec'):
                y_dataset_eval_train_loss.append(float(result[2]))
                # print(y_dataset_eval_train_loss)
                pattern = re.compile(r'-?[0-9]\d*\.?\d*')  # 查找数字
                result = pattern.findall(line)
                x_dataset_eval_epoch.append(float(result[1]))
                y_dataset_eval_loss.append(float(result[2]))
                eval_rrse.append(float(result[3]))
        except Exception as e:
            pass

    plt.figure(figsize=(11,10))
    ax = plt.subplot(4, 1, 1)
    plt.plot(x_dataset_time, y_dataset_train_loss, color='red', label="train_loss")
    plt.plot(x_dataset_time, y_dataset_kl_loss, color='blue', label="kl_loss")
    plt.plot(x_dataset_time, y_dataset_likelihood_loss, color='black', label="likelihood_loss")
    plt.xlabel('Time(s)')
    ax.set_ylabel('loss')
    ax.set_title('train_loss')
    # plt.xticks(())
    # plt.yticks(())
    plt.legend()
    # plt.savefig(os.path.join(base_dir, 'train_loss.png'))

    ax = plt.subplot(4, 1, 2)
    plt.plot(x_dataset_eval_epoch, y_dataset_eval_loss, 'g-', label='eval_loss')
    plt.plot(x_dataset_eval_epoch, y_dataset_eval_train_loss, 'r-', label='train_loss')
    plt.legend()
    plt.xlabel('epochs')
    ax.set_ylabel('eval_loss')
    ax.set_title('eval_loss')

    ax = plt.subplot(4, 1, 3)
    # plt.plot(x_dataset_eval_epoch, y_dataset_eval_loss, 'g-', label='eval_loss')
    plt.plot(x_dataset_eval_epoch, eval_rrse, 'r-', label='rrse')
    plt.legend()
    plt.xlabel('epochs')
    ax.set_ylabel('eval_rrse')
    ax.set_title('eval_rrse')

    ax = plt.subplot(4, 1, 4)
    # plt.plot(x_dataset_eval_epoch, y_dataset_eval_loss, 'g-', label='eval_loss')
    plt.plot(x_dataset_eval_epoch, eval_rrse, 'r-', label='lr')
    plt.legend()
    plt.xlabel('epochs')
    ax.set_ylabel('lr')
    ax.set_title('learning rate')

    plt.tight_layout()

    plt.savefig(os.path.join(base_dir, 'train_visualize.png'))

    plt.close()  # 关闭图像，避免出现warning
    # plt.plot(x_dataset_time,y_dataset_train_loss,color='red',label="train_loss")
    # plt.plot(x_dataset_eval_epoch,y_dataset_eval_loss,color='black',label="loss")


if __name__ == '__main__':
    base_dir = '/root/data/SE-VAE/ckpt/winding/ct_True/ode_rssm_schedule/ode_rssm_ct_time=True,model.iw_trajs=1,model.k_size=16,model.ode_ratio=all,model.ode_solver=rk4,model.ode_type=orth,model.state_size=16,random_seed=0,sp=0.5,train.batch_size=512,train/schedule=val_step/2022-05-04_07-49-29'
    training_loss_visualization(base_dir)

