#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import numpy as np
import pandas as pd
import random
import os
import copy
from scipy import signal
from matplotlib import pyplot as plt
def download(url, path):
    import urllib
    print("Downloading file %s from %s:" % (path, url))
    try:
        urllib.request.urlretrieve(url, filename=os.path.join( path))
        return path
    except Exception as e:
        print("Error occurred when downloading file %s from %s, error message :" % (path, url))
        return None


def detect_download():
    import pandas as pd
    data_urls = pd.read_csv('../data/data_url.csv')
    base = '../data'
    data_paths = []
    for name, url in zip(data_urls['object'], data_urls['url']):
        path = os.path.join(base, 'part', name)
        if not os.path.exists(path) and not download(url, path):
            pass
        else:
            data_paths.append(path)
    return data_paths


def data_filter(csv_list,filter_mode, names):
    def plot_cmp(old, new, csv_index, series_index, pos):
        plt.subplot(1,5,pos)
        plt.plot(old)
        plt.plot(new)
        plt.legend(['old', 'new'])
        plt.title(str(csv_index)+'-'+id2name[series_index])

    def csv_smoother(csv, index, plot=False):
        cmp_series = []
        if filter_mode == 'med':
            for c in names:
                old_series = np.array(csv[c])
                csv[c]=signal.medfilt(csv[c],kernel_size=9)# kernel_size是滤波直径
                new_series = np.array(csv[c])
                cmp_series.append((old_series, new_series))
        else:
            raise NotImplementedError
        # 画图对比滤波效果，每个csv一行5张子图
        if plot:
            plt.figure(figsize=(15,3))
            for i, (old, new) in enumerate(cmp_series):
                 plot_cmp(old, new, index, list(names)[i],i+1)
            plt.show()
            plt.close()
        return csv
    csv_list = [csv_smoother(csv, i+1) for (i, csv) in enumerate(csv_list)]
    return csv_list


def generate_episode(
        csv_list,
        name2id,
        min_dis=10,
        max_dis=30,
        aggregate='sum',
        filter_mode='med'
):
    # 对每个df中的数据进行滤波
    csv_list = data_filter(csv_list, filter_mode, name2id.values())

    def solve(csv, aggregate):
        count = len(csv) // 5
        start = np.sort(np.random.randint(0, len(csv) - max_dis, size=count))

        end = start + np.random.randint(min_dis, max_dis, size=count)
        assert (start < end).all()

        def feature_maker(df, name2id, aggregate):

            if aggregate == 'sum':
                aggregate = np.sum
                base = 1
            elif aggregate == 'mean':
                aggregate = np.mean
                base = df.shape[0]
            else:
                raise NotImplementedError
            v_in = np.array(df[name2id['v_in']], dtype=np.float32)
            v_out = np.array(df[name2id['v_out']], dtype=np.float32)
            c_in = np.array(df[name2id['c_in']], dtype=np.float32) / 100
            c_out = np.array(df[name2id['c_out']], dtype=np.float32) / 100
            pressure = np.array(df[name2id['pressure']], dtype=np.float32)
            det_v = aggregate(v_in) - aggregate(v_out)
            det_vc = aggregate(v_in * c_in) - aggregate(v_out * c_out)
            det_vc2 = aggregate(v_in * c_in ** 2) - aggregate(v_out * c_out ** 2)
            det_vc3 = aggregate(v_in * c_in ** 3) - aggregate(v_out * c_out ** 3)
            det_pressure = (pressure[-1] - pressure[0])/base
            return np.array([
                det_v,
                det_vc,
                det_vc2,
                det_vc3,
                det_pressure
            ], dtype=np.float32)

        result_np = np.stack([feature_maker(csv[a:b], name2id, aggregate) for (a, b) in zip(start, end)], axis=0)
        #         import pdb
        #         pdb.set_trace()
        return result_np

    labels = [
        'det_v',
        'det_vc',
        'det_vc2',
        'det_vc3',
        'det_pressure'
    ]
    return np.concatenate([solve(csv, aggregate) for csv in csv_list], axis=0), labels


data_paths = detect_download()
name2id = {
    'v_in':'16',
    'v_out':'14',
    'c_in':'4',
    'c_out':'11',
    'pressure':'17',
}
id2name = {v:k for (k,v) in name2id.items()}
csv_list = [pd.read_csv(file) for file in data_paths]

segment_length = np.arange(1, 100, 5)
parameters_group_num = 3000
points_num = 1000


pearsonr_df = pd.DataFrame(columns=['det_v','det_vc','det_vc2','det_vc3','det_pressure'], index=segment_length)
for l in segment_length:
    print('length: %d' % l)
    made_array, labels = generate_episode(csv_list,
                     name2id,
                     min_dis=l,
                     max_dis=l+1,aggregate='sum',
                     filter_mode='med')
    df = pd.DataFrame(made_array, columns=labels, index=range(0,len(made_array)))
    print(df.describe())
    from scipy.stats import pearsonr
    for item in labels:
        pear = pearsonr(df[item], df['det_pressure'])
        pearsonr_df.loc[l][item] = pear[0]
        print(item, pear)

    coef_intercept_list = []
    for group in range(parameters_group_num):
        begin_pos = np.random.randint(0, len(df)-points_num)
        train_x, train_y = made_array[begin_pos: begin_pos+points_num, :-1], made_array[begin_pos: begin_pos+points_num, -1:]

        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(train_x, train_y)
        coef_intercept_list.append(np.concatenate([reg.coef_.squeeze(), reg.intercept_]))

    writen_df = pd.DataFrame(data=np.stack(coef_intercept_list),
                             columns=['coef_'+x for x in ['det_v', 'det_vc','det_vc2', 'det_vc3']] + ['intercept'])
    writen_df.to_csv(os.path.join('../data/linear', str(l)+'.csv'))

pearsonr_df.to_csv('./pear.csv')


