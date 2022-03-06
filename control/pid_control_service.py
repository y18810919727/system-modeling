#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import os
import json


import torch
from flask import Flask, jsonify, request
import sys
from control.algorithms.pid_planning import PIDPlanning
import time
from control.utils import dict_to_Tensor, my_JSON_serializable, DictConfig2dict
import hydra
from omegaconf import DictConfig
sys.path.append('..')
sys.path.append(os.getcwd())

app = Flask(__name__)

controller_info = {
    'device': None,
    'args': None,
    'pressure_value': args.pressure_value,
    'scale': None
}


def get_basic_info():
    global controller_info
    return {
        'device': str(controller_info['device']),
        'model_config': DictConfig2dict(controller_info['args'])
    }


@app.route('/test', methods=['GET'])
def test_request():
    if request.method == 'GET':
        return jsonify('Received!')


@app.route('/planning', methods=['POST'])
def pid_planning():     # pid控制
    if request.method == 'POST':

        global controller_info
        memory_state = request.form['memory_state']
        memory_state = json.loads(memory_state)
        memory_state = dict_to_Tensor(memory_state, device=controller_info['device'])

        if 'time' in request.form.keys():
            systime = request.form['time']
        else:
            systime = -1
        args = controller_info['args']

        begin_time = time.perf_counter()
        # input_dim: 2, output_dim: 1

        if 'set_point' in request.form.keys() and request.form['set_point'] is not None:
            set_point = json.loads(request.form['set_point'])
        else:
            set_point = args['default_set_point']

        if args.algorithm.name == 'pid':
            pid_config = args.algorithm
            planner = PIDPlanning(pid_config.dt, args.max, args.min, pid_config.KP, pid_config.KI, pid_config.KD, time=systime)
        else:
            raise NotImplementedError

        if 'set_point' in request.form.keys() and request.form['set_point'] is not None:
            set_point = json.loads(request.form['set_point'])
        else:
            set_point = args['default_set_point']

        action = planner.solve(set_point, args.pressure_value)
        # action_f 为cem loss最小的动作序列的均值
        # controller_info[''] = new_dist
        # TODO: 不要出现for循环
        # action_sample = [float(action_f[x]) for x in range(0, len(action_f))]   # action_f转numpy
        # u_sample = [float(action_f[x]) for x in range(0, len(action_f))]        # u_f转numpy
        time_used = time.perf_counter() - begin_time

        response_dict = {
            'planning_action': action,   # action_f转list //v_out
            'time_usage': '{}s'.format(time_used),
        }
        response_dict.update(get_basic_info())
        return jsonify(response_dict)


@app.route('/update', methods=['POST'])
def update():      # 更新隐状态
    print('Post received')
    if request.method == 'POST':

        begin_time = time.perf_counter()    # 当前计算机系统时间

        new_memory_state = None
        time_used = time.perf_counter() - begin_time  # 记录程序执行时间

        response_dict = {
            'memory_state': my_JSON_serializable(new_memory_state),
            'time_usage': '{}s'.format(time_used),
        }
        response_dict.update(get_basic_info())    # 更新字典（键值对）
        return jsonify(response_dict)


@hydra.main(config_path='config', config_name='config_pid.yaml')
def control_service_start(args: DictConfig):

    if args.cuda >= torch.cuda.device_count():
        raise RuntimeError('cuda %s invalid device ordinal' % args.cuda)

    device = torch.device("cuda:{}".format(str(args.cuda)) if torch.cuda.is_available() and args.cuda != -1 else "cpu")

    global controller_info
    # 向controller_info中添加模型或其他参数
    controller_info['device'] = device
    controller_info['args'] = args
    print(args)
    # controller_info['scale'] = Scale(mean=np.zeros(5), std=np.ones(5))  # 此处需要利用数据集进行估计，应保持和训练时的归一化一致
    # df_scale    [c_out, v_out, c_in, v_in, pressure]

    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':

    control_service_start()
