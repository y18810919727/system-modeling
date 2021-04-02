# 泥层压力控制代码说明


## 启动方式

> 当前目录需要设定为项目根目录下，如```/home/yuanzhaolin/SE-VAE/```

1. 先启动flask server
```python
python control/pressure_control_service.py --debug
```
2. 启动pressure_test

```python
python control/pressure_control_service.py --debug
```

### 需要实现的内容
1. cem_planning.py
    - 定义惩罚函数 eval()
    - cem过程 solve()
2. thickener_pressure_simulation.py
    - 利用模型进行浓密机仿真
3. 控制结果的可视化：画图，算指标


