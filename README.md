# SE-VAE

1. 如果使用仿真数据测试，先运行```data/fake_data_generator.py```生成仿真数据，如果使用主西矿体数据，青葱
2. 运行```model_train.py```训练模型
如：

```python
CUDA_VISIBLE_DEVICES=3 python model_train.py --dataset fake --use_cuda --save_dir myself --version 1
```
 使用3号GPU，其他参数默认，训练模型与日志存储在ckpt/fake_v1_myself中。
 
3. 运行```model_test``` 测试模型
```python
CUDA_VISIBLE_DEVICES=3 python model_test.py --dataset fake --use_cuda --save_dir myself --version 1 
```
自动检测```best.pth```模型，其他参数需要与训练时的参数保持一致(包括模型版本、网络节点数等等)，测试之后会在目录下生成```figs```文件夹以及```test.out```日志文件

4. 部分参数介绍：
- ```dataset```: ```fake``` 或 ```west```是，设定为```west```时自动从aliyun的oss下载数据文件,如果使用仿真数据测试，如果设定为```fake```，则采用仿真数据，需要先运行```data/fake_data_generator.py```生成仿真数据，
- ```L``` 决定采样次数，越大训练结果越准但是耗时越长，仅对与v1和v5有影响
- ```version``` 选择算法模型，

    > 目前包含的版本如下:

    | 版本名      |   模型   |   备注   |
    | :-------- | --------:| :------: |
    | v1    |   VAE，生成数据的似然靠估计 |   效果与v2接近 |
    | v2    |   VAE，生成数据的似然靠解析计算 |  目前所有实验效果中最好  |
    | v3    |   用ELBO训练自适应卡尔曼滤波|  有点慢  |
    | v4    |   用数据似然训练自适应卡尔曼滤波|   有点慢 |
    | v5    |   隐变量到观测变量的过程用lstm建模|  效果很差，应该是采样方差太大的原因  |
- ```plt_cnt```: 决定test时会画几个图在figs中
- ```save_dir```: 个性设置最后模型以及日志的保存目录，具体形式为```\{dataset\}\_v\{version\}\_\{save_dir\}```，如上面例子中的```fake_v1_myself```

