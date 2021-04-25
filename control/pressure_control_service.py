#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json


import torch
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
def cem_planning():
    if request.method == 'POST':

        global controller_info
        memory_state = request.form['memory_state']
        memory_state = json.loads(memory_state)
        memory_state = dict_to_Tensor(memory_state, device=controller_info['device'])

        args = controller_info['args']

        begin_time = time.perf_counter()
        # input_dim: 1, output_dim: 4
        cem = CEMPlanning(1, 4, args.length, num_samples=args.length, max_iters=args.max_iters, device=controller_info['device'])
        new_dist = cem.solve(controller_info['model'], memory_state, last_seq_distribution=controller_info['last_seq_distribution'])

        controller_info['last_seq_distribution'] = new_dist
        action_sample = new_dist.sample()[0].detach().cpu().numpy()  # 从cem求得控制序列采样，并取第一个
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
def update():
    if request.method == 'POST':

        global controller_info
        if 'memory_state' in request.form.keys() and request.form['memory_state'] is not None:
            memory_state = request.form['memory_state']
            memory_state = json.loads(memory_state)
            for k, v in memory_state.items():
                memory_state[k] = torch.Tensor(
                    np.array(v if isinstance(v, list) else json.loads(v), dtype=np.float32),
                ).to(controller_info['device'])
        else:
            memory_state = None

        new_monitoring_data = request.form['monitoring_data']
        new_monitoring_data = json.loads(new_monitoring_data)
        begin_time = time.perf_counter()

        monitoring_data_scale = controller_info['scale'].scale_array(np.array(new_monitoring_data))  # 数据归一化
        v_out = monitoring_data_scale[:, -1]
        observation = monitoring_data_scale[:, :-1]
        seq_length = observation.shape[0]

        # region TODO: 目前的模型是8输入，1输出, 现在为了测试先把最后一个维度当作输出，未来需要替换为下面两行
        #
        # external_input = torch.Tensor([v_out]).to(controller_info['device']).reshape(1, 1, -1)
        # observations_seq = torch.Tensor(observation).to(controller_info['device']).reshape(1, 1, -1)
        external_input = torch.Tensor(observation).to(controller_info['device']).reshape(seq_length, 1, -1)
        observations_seq = torch.Tensor([v_out]).to(controller_info['device']).reshape(seq_length, 1, -1)
        # endregion

        _, _, new_memory_state = controller_info['model'].forward_posterior(
            external_input, observations_seq, memory_state=memory_state
        )

        time_used = time.perf_counter() - begin_time  # 记录程序执行时间

        response_dict = {
            'memory_state': my_JSON_serializable(new_memory_state),
            'time_usage': '{}s'.format(time_used),
        }
        response_dict.update(get_basic_info())
        return jsonify(response_dict)


def control_service_start(args):

    # import sys
    # sys.stderr = open('control/logs/service_log.out', 'w')

    # torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda:{}".format(str(args.cuda)) if torch.cuda.is_available() and args.cuda != -1 else "cpu")
    model = torch.load(args.planning)
    model = model.to(device)
    model.eval()

    global controller_info
    # 向controller_info中添加模型或其他参数
    controller_info['device'] = device
    controller_info['model'] = model
    controller_info['args'] = args
    controller_info['scale'] = Scale(mean=np.zeros(9), std=np.ones(9))  # 此处需要利用数据集进行估计，应保持和训练时的归一化一致
    from pprint import pprint
    pprint(controller_info)
    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Pressure Control Test')
    parser.add_argument('--planning',  type=str, default='control/model/test.pkl', help="ckpt path of planning model.")
    parser.add_argument('--cuda',  type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--length',  type=int, default=50, help="The length of optimized sequence for planning")
    parser.add_argument('--num_samples',  type=int, default=32, help="The number of samples in CEM planning")
    parser.add_argument('--num_iters',  type=int, default=32, help="The number of iters in CEM planning")
    parser.add_argument('--max_iters',  type=int, default=50, help="The number of iters in CEM planning")
    parser.add_argument('--port',  type=int, default=6008, help="The number of iters in CEM planning")
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    control_service_start(args)
