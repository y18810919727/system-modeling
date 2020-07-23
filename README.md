# SE-VAE

1. 先运行```data/fake_data_generator.py```生成仿真数据
2. 运行```model_train.py```训练模型
如：
```python
CUDA_VISIBLE_DEVICES=3 python model_train.py --dataset fake --use_cuda --save_dir myself
```
> 使用3号GPU，其他参数默认，训练模型与日志存储在ckpt/myself_v1中，其中v1为算法版本，未来会开发v2，v3，训练时通过--version参数进行指定。

3. 运行```model_test``` 测试模型
```python
CUDA_VISIBLE_DEVICES=3 python model_test.py --dataset fake --use_cuda --test_id 599
```
其中```test_id```为待测试的pth模型名字，其他参数默认
