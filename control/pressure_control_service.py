#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json


import torch
import pandas as pd
from flask import Flask
import getopt
from flask import request
from flask import Flask, jsonify, request
import sys
sys.path.append('..')
sys.path.append(os.getcwd())
from control.cem_planning import CEMPlanning
from control.scale import Scale
import time
from control.utils import dict_to_Tensor, my_JSON_serializable
import argparse
app = Flask(__name__)

import logging

# 此处load模型
controller_info = {
    'model': None,
    'device': None,
    'args': None,
    'last_seq_distribution': None,
    'scale': None
}


def get_basic_info():
    global controller_info
    return {
        'device': str(controller_info['device']),
        'model_config': controller_info['args'].__dict__
    }


@app.route('/test', methods=['GET'])
def test_request():
    if request.method == 'GET':
        return jsonify('Received!')


@app.route('/cem-planning', methods=['POST'])
def cem_planning():     # Cem规划控制
    if request.method == 'POST':

        global controller_info
        memory_state = request.form['memory_state']
        memory_state = json.loads(memory_state)
        memory_state = dict_to_Tensor(memory_state, device=controller_info['device'])

        systime = request.form['time']
        args = controller_info['args']

        begin_time = time.perf_counter()
        # input_dim: 1, output_dim: 4
        #num_samples = args.length
        cem = CEMPlanning(1, 4, args.length, num_samples=args.num_samples, max_iters=args.max_iters, device=controller_info['device'], time=systime)
        new_dist, action_f = cem.solve(controller_info['model'], memory_state, last_seq_distribution=controller_info['last_seq_distribution'])
        # action_f 为cem loss最小的动作序列的均值
        controller_info['last_seq_distribution'] = new_dist
        action_sample = action_f.detach().cpu().numpy()  # 从cem求得控制序列采样，并取第一个
        time_used = time.perf_counter() - begin_time

        # 反归一化底流流量, 对照一下出料流量是scale的哪一列
        action_sample_unscale = controller_info['scale'].unscale_scalar(action_sample, pos=-1)
        response_dict = {
            'planning_v_out': action_sample_unscale.tolist(),
            'time_usage': '{}s'.format(time_used),
        }
        response_dict.update(get_basic_info())
        return jsonify(response_dict)


@app.route('/predict', methods=['POST'])
def predict_forward():
    """
    可以用于系统预测，尚未实现，日后待补
    Returns:

    """
    raise NotImplementedError


@app.route('/update', methods=['POST'])
def update():      # 更新隐状态
    print('Post received')
    if request.method == 'POST':

        global controller_info
        if 'memory_state' in request.form.keys() and request.form['memory_state'] is not None:
            memory_state = request.form['memory_state']
            memory_state = json.loads(memory_state)
            memory_state = dict_to_Tensor(memory_state, device=controller_info['device'])
        else:
            memory_state = None

        new_monitoring_data = request.form['monitoring_data']
        new_monitoring_data = json.loads(new_monitoring_data)
        begin_time = time.perf_counter()    # 当前计算机系统时间

        monitoring_data_scale = controller_info['scale'].scale_array(np.array(new_monitoring_data, dtype=float))  # 数据归一化
        # monitoring_data_scale = monitoring_data_scale.tolist()
        v_out = monitoring_data_scale[-1]     # 底流流量在倒数第一个位置，输入
        observation = monitoring_data_scale[:-1]     # 输出

        # region TODO: 目前的模型是8输入，1输出, 现在为了测试先把最后一个维度当作输出，未来需要替换为下面两行
        #
        # external_input = torch.Tensor([v_out]).to(controller_info['device']).reshape(1, 1, -1)
        # observations_seq = torch.Tensor(observation).to(controller_info['device']).reshape(1, 1, -1)
        external_input = torch.Tensor([v_out]).to(controller_info['device']).reshape(1, 1, -1)  # 重组、降维
        observations_seq = torch.Tensor(observation).to(controller_info['device']).reshape(1, 1, -1)
        # endregion

        _, _, new_memory_state = controller_info['model'].forward_posterior(
            external_input, observations_seq, memory_state=memory_state
        )

        time_used = time.perf_counter() - begin_time  # 记录程序执行时间

        response_dict = {
            'memory_state': my_JSON_serializable(new_memory_state),
            'time_usage': '{}s'.format(time_used),
        }
        response_dict.update(get_basic_info())    # 更新字典（键值对）
        return jsonify(response_dict)


def control_service_start(args):

    # import sys
    # sys.stderr = open('control/logs/service_log.out', 'w')

    # torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda:{}".format(str(args.cuda)) if torch.cuda.is_available() and args.cuda != -1 else "cpu")
    model = torch.load(args.planning, map_location={'cuda:0': 'cuda:{}'.format(str(args.cuda)) if torch.cuda.is_available() and args.cuda != -1 else "cpu"})
    model = model.to(device)
    model.eval()

    global controller_info
    # 向controller_info中添加模型或其他参数
    controller_info['device'] = device
    controller_info['model'] = model
    controller_info['args'] = args
    # 修改为 1输入 - 4输出   即 9 -> 5
    # [pressure, c_out, v_in, c_in, v_out]['17', '11', '16', '4', '14']
    # controller_info['scale'] = Scale(mean=np.zeros(5), std=np.ones(5))  # 此处需要利用数据集进行估计，应保持和训练时的归一化一致
    # df_scale    [c_out, v_out, c_in, v_in, pressure]
    df_scale = pd.read_csv(os.path.join(os.getcwd(), "control/df_scale_cstr.csv"), encoding='utf-8').values   # dataframa 转array
    mean = np.array(df_scale[:, 1], dtype='float64')
    std = np.array(df_scale[:, 2], dtype='float64')
    controller_info['scale'] = Scale(mean, std)
    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Pressure Control Test')
    parser.add_argument('--planning',  type=str, default='control/model/test_west.pkl', help="ckpt path of planning model.")
    parser.add_argument('--cuda',  type=int, default=3, help="GPU ID, -1 for CPU")
    parser.add_argument('--length',  type=int, default=50, help="The length of optimized sequence for planning")
    parser.add_argument('--num_samples',  type=int, default=32, help="The number of samples in CEM planning")
    parser.add_argument('--num_iters',  type=int, default=32, help="The number of iters in CEM planning")
    parser.add_argument('--max_iters',  type=int, default=50, help="The number of iters in CEM planning")
    parser.add_argument('--port',  type=int, default=6009, help="The number of iters in CEM planning")
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    control_service_start(args)
