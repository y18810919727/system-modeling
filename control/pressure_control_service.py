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
        # input_dim: 2, output_dim: 1
        cem = CEMPlanning(args.set_value, args.input_dim, args.output_dim, args.length, num_samples=args.num_samples, max_iters=args.max_iters, device=controller_info['device'], time=systime)
        new_dist, action_f = cem.solve(controller_info['model'], memory_state, last_seq_distribution=controller_info['last_seq_distribution'])
        # action_f 为cem loss最小的动作序列的均值
        controller_info['last_seq_distribution'] = new_dist
        action_sample = [action_f[x].detach().cpu().numpy() for x in range(0, len(action_f))]   # action_f转numpy
        time_used = time.perf_counter() - begin_time

        response_dict = {
            'planning_action': [action_sample[x].tolist() for x in range(0, len(action_sample))],   # action_f转list //v_out
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

        action = np.array(new_monitoring_data[-1])
        observation = np.array(new_monitoring_data[0])
        external_input = torch.Tensor(action).to(controller_info['device']).reshape(1, 1, -1)  # 重组、降维
        observations_seq = torch.Tensor(observation).to(controller_info['device']).reshape(1, 1, -1)
        # endregion

        _, new_memory_state = controller_info['model'].forward_posterior(
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
    # controller_info['scale'] = Scale(mean=np.zeros(5), std=np.ones(5))  # 此处需要利用数据集进行估计，应保持和训练时的归一化一致
    # df_scale    [c_out, v_out, c_in, v_in, pressure]

    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Pressure Control Test')
    parser.add_argument('--planning',  type=str, default='control/model/cstr_vrnn_5.pkl', help="ckpt path of planning model.")
    parser.add_argument('--cuda',  type=int, default=3, help="GPU ID, -1 for CPU")
    parser.add_argument('--length',  type=int, default=50, help="The length of optimized sequence for planning")
    parser.add_argument('--num_samples',  type=int, default=32, help="The number of samples in CEM planning")
    parser.add_argument('--num_iters',  type=int, default=32, help="The number of iters in CEM planning")
    parser.add_argument('--max_iters',  type=int, default=50, help="The number of iters in CEM planning")
    parser.add_argument('--input_dim', type=int, default=1, help='The input_dim of model')
    parser.add_argument('--output_dim', type=int, default=2, help='The output_dim of model')
    parser.add_argument('--set_value', type=list, default=[0.8,0.1,0.5], help='The set_value of control')  # 0.8[number of output_dim; number of input_dim]
    parser.add_argument('--port',  type=int, default=6010, help="The number of iters in CEM planning")
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    control_service_start(args)
