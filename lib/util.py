#!/usr/bin/python
# -*- coding:utf8 -*-

import os
import yaml
import numpy as np
import json
# from ruamel_yaml import
import time
import os
import re

from pandas import DataFrame
from omegaconf import DictConfig, OmegaConf
from functools import reduce
from operator import and_, or_


def load_DictConfig(path, name):
    file = os.path.join(
        path, name
    )
    if not os.path.exists(os.path.join(path, name)):
        return None

    config = OmegaConf.load(file)

    return DictConfig(config)


def write_DictConfig(path, name, exp_dict: DictConfig):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, name), "w", encoding="utf-8") as f:
        OmegaConf.save(exp_dict, f)


def load_yaml(path, name):

    file = os.path.join(
        path, name
    )
    if not os.path.exists(os.path.join(path, name)):
        return None

    with open(file, "r", encoding="utf-8") as f:
        config = yaml.load(f)

    return DictConfig(config)


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

# TODO: jupyter 数据总表移植


def my_filter(path):
    log_path = os.path.join('..', path, 'log.out')
    s = open(log_path, 'r').readlines()
    log_path = os.path.join('..', path, 'test.out')
    test_s = open(log_path, 'r').readlines()
    if False:
        return False
    elif 'Error' in test_s[-2]:
        return False
    else:
        return True


def generating_dir(base_dir, root_dir):
    ret = []
    for file in os.listdir(os.path.join(root_dir, base_dir)):
        if file == 'tmp':
            continue
        path = os.path.join(base_dir, file)
        try:
            if os.path.isdir(os.path.join(root_dir, path)):
                ret = ret + generating_dir(path, root_dir)
            elif file == 'test.out' and os.path.exists(os.path.join(root_dir, base_dir, 'best.pth')):
                # print(path)
                if my_filter(os.path.split(path)[0]):
                    ret.append(os.path.split(path)[0])
                else:
                    continue
        except Exception as e:
            print(f'Generating dir {path} failed')
            continue

    return ret


def generate_data_frame(ckpt_dir='ckpt', root_dir='../', date=None, model=None, key_words=None,
                        dataset=None, sort_key='pred_rmse'):
    """

    Args:
        date: only counting the logs later than date
        model:  save_dir
        ckpt_dir: the name of ckpt dir
        root_dir: the position of ckpt_dir

    Returns:
        data_frames: List of Dataframe. {Dataframe-1, Dataframe-2, ... }
        datasets: List of str. {datset-1, datset-2 ,...}
        path_list: List of str, all paths of ckpt

    How to use:

        datasets, data_frames, path_list = generate_data_frame(ckpt_dir='ckpt', root_dir='../', sort_key='likelihood',
            dataset='winding', sort_key='pred_rrse', key_words=['rssm', 'vrnn'])
    """
    if key_words is None:
        key_words=['/']
    if isinstance(key_words, str):
        key_words = [key_words]

    if isinstance(model, str):
        model = [model]

    path_list = generating_dir(ckpt_dir, root_dir)
    date_filter = lambda path: True if date is None else path.split('/')[-1] >= date
    model_filter = lambda path: True if model is None else path.split('/')[2] in model
    dataset_filter = lambda path: True if dataset is None else path.split('/')[1] == dataset
    key_words_filter = lambda path: True if key_words is None else reduce(and_, [kw in path for kw in key_words])

    path_list = list(filter(lambda x: reduce(and_, [
        date_filter(x),
        model_filter(x),
        dataset_filter(x),
        key_words_filter(x),
    ]), path_list))

    data = list(set([path.split('/')[1] for path in path_list]))
    # 获取数据集/模型名称及个数
    print(data)
    # data = data[:-1]

    df = []  # DataFrame集合
    # x=np.zeros((n_dex,n_col))#数值矩阵
    for d in range(len(data)):
        temp_list = []  # 存放单个数据集的临时列表
        dex = []  # 行列表
        col = []  # 列列表
        # print(d)
        for path in path_list:
            if path.split('/')[1] == data[d]:
                temp_list.append(path)
        # print(temp_list)
        for path in temp_list:
            # pattern=re.compile(r'\/.*?/')
            result = path.split('/')
            # print(result)
            dex.append(result[len(result) - 2] + '//' + result[len(result) - 1])
        # print('dex', len(dex))
        n_dex = len(dex)  # 行数
        # print(dex)
        # assert len(dex) == len(temp_list)

        # 单独开启第一个path文件获取列数以便于初始化数值矩阵
        f = open('../' + temp_list[0] + '/test.out', 'r')
        temp_data = f.readlines()
        f.close()

        col2id = {}
        for line in temp_data:
            if re.search('likelihood', line):
                # pattern = re.compile(r'-?[0-9]\d*\.\d*')   #查找数字(之查找包含小数点的)
                t_col = re.findall(r'(\w*\(?\w\)?\w*)=(-?[0-9]\d*\.\d*)', line)
                # print(t_col)
                for i in range(len(t_col)):
                    if t_col[i][0] == 'time':
                        pass
                    else:
                        col.append(t_col[i][0])
                        col2id[t_col[i][0]] = len(col) - 1
        # print('col', len(col))
        n_col = len(col)  # 列数
        x = np.zeros((n_dex, n_col))  # 数值矩阵
        for t in range(len(temp_list)):  # 第t个临时列表中的路径
            try:
                # print('t=',t)
                f = open('../' + temp_list[t] + '/test.out', 'r')
                temp_data = f.readlines()
                f.close()
                # 数值矩阵生成
                for line in temp_data:
                    if re.search('likelihood', line):
                        # print(line[11:len(line)])
                        # print(len(line))
                        pattern = re.compile(r'-?[0-9]\d*\.\d*')  # 查找数字(之查找包含小数点的)
                        result = pattern.findall(line)
                        # 数据对应填入该行
                        t_x = re.findall(r'(\w*\(?\w\)?\w*)=(-?[0-9]\d*\.\d*)', line)
                        # print(t_x, len(t_x))
                        # print(t_x)
                        for i in range(len(t_x)):
                            if t_x[i][0] == 'time':
                                pass
                            else:
                                # print(x.shape, t, i)
                                if t_x[i][0] in col2id.keys():
                                    x[t, col2id[t_x[i][0]]] = t_x[i][1]  # 不同ckpt的统计指标有可能不一致
                                # x[t, i] = t_x[i][1]  # 由于index填入模型名称的顺序就是index列表的序号，因此行与list一一对应
            except Exception as e:
                print(temp_list[t], ' is not identificated')
                raise e
                    # print(x)
        # 生成DataFrame
        # print(col)
        df.append(DataFrame(x, columns=col, index=dex).sort_values(by=sort_key))
        # 当前数据集frame生成完毕，临时列表清零

    return data, df, path_list


if __name__ == '__main__':
    # data, dfs, path_list = generate_data_frame(ckpt_dir='ckpt', root_dir='../', model='oderssm_ode')
    data, dfs, path_list = generate_data_frame(ckpt_dir='ckpt', root_dir='../', sort_key='likelihood', key_words=['rssm', 'vrnn'])
    print('\n'.join(path_list))
    # for df in dfs:
    #     print(pa)
