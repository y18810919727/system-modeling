# SE-VAE

## 模型训练

- multirun：根据参数的笛卡尔积同时训练多个模型，
``` python
CUDA_VISIBLE_DEVICES=3 python model_train.py --multirun model.D=1,5,10,15,20 dataset=west model.k_size=16 model.dynamic.num_linears=8
```
 使用3号GPU，参数D分别为1,5,10,15,20五种情况,训练五个模型，其他参数默认。

> 参考```scripts/vrnn_multirun.sh```文件

 - 单模型训练：不加multirun参数

``` python
CUDA_VISIBLE_DEVICES=3 python model_train.py train.batch_size=32 dataset=winding model.D=1
```
训练模型与日志存储在```ckpt/${dataset.type}/${save_dir}/${model.type}_${实验参数}/${now:%Y-%m-%d_%H-%M-%S}```中

训练完成后将自动绘制loss图，并在测试集上进行评估，目录包含的文件：测试之后会在目录下生成```figs```文件夹以及```test.out```日志文件
- figs: 测试集中部分数据的预测可视化结果，可修改test.plt_cnt
- best.pth: 验证集score最高的模型参数
- log.out: 训练过程日志
- train_loss.png: 训练loss图
- val_loss.png: 验证集loss和训练集loss对比图
- test.out: 验证集中每条数据对的预测指标详情

## 手动测试
运行```model_test``` 手动测试模型
``` python
CUDA_VISIBLE_DEVICES=3 python model_test.py 'save_dir=ckpt/west/vrnn/vrnn_model.D\=3/2020-12-03_07-20-37' dataset=winding model.k_size=16 model.dynamic.num_linears=8 model.D=25
```
> 配置测试目录为save_dir=xxx时，对于xxx中的特殊符号，包括'='或','等，需要在前面加单斜杠'\\'转义。

运行后将自动在当前目录下生成```figs```文件夹以及```test.out```日志文件。

> 参考```scripts/srnn_multirun.sh```文件

## 部分参数介绍：
- ```dataset```：```cstr, west, winding```，```data```目录下无数据时自动从aliyun的oss下载数据文件。

- ```ct_time```:对数据机随机下采样，使得数据时间间隔非均匀，配合 ```sp```(下采样比率)使用。开启后，external_input会自动增加一维

- ```D```:：overshooting最大距离

- ```model.state_size``` ：latent state的大小

- ```version``` ：选择算法模型，
    
    > 目前包含的版本如下:
    
    | 版本名      |   模型   |   备注   | 参考文献 |
    | :-------- | --------:| :------: | --------- |
    | vaecl   |  VAE combinational linears  |   动态系统为自适应线性组合，线性解器 | 部分参考：Fraccaro, M., Kamronn, S., Paquet, U., & Winther, O. (2017). A disentangled recognition and nonlinear dynamics model for unsupervised learning. *Advances in Neural Information Processing Systems*, *2017*-*Decem*(section 5), 3602–3611. |
    | vrnn | variational RNN |  | Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A., & Bengio, Y. (2015). A recurrent latent variable model for sequential data. *Advances in Neural Information Processing Systems*, *2015*-*Janua*, 2980–2988. |
    | srnn | stochastic recurrent neural network |  | Fraccaro, M., Sønderby, S. K., Paquet, U., & Winther, O. (2016). Sequential neural models with stochastic layers. *Advances in Neural Information Processing Systems*, 2207–2215. |
    | seq2seq | attention seq2seq |  |References: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html |
    
- ```plt_cnt```: 决定运行```model_test.py```时会画几个图在figs中

## jupyter脚本
- ```jupyter/generate_data_frame.ipynb```：生成所有数据集的所有模型的对比结果(不考虑```ckpt/${dataset.type}/tmp/``` 目录下的模型)
- ```jupyter/ErrorLog.ipynb```: 自动检测并删除训练失败的日志目录(不考虑```ckpt/${dataset.type}/tmp/``` 目录下的模型)
## hydra  
参考资料: https://hydra.cc/


## 控制部分的Docker build

1. 构建镜像
```bash
docker build -t ai3d/uc_control:v1 . 
```
2. 启动容器
```bash
docker run --rm --gpus all -it -p 6010:6010 ai3d/uc_control:v1 cuda=0
```
参数解释:
- cuda: 使用GPU ID ，不加参数默认为0号gpu
- 其他参数见```control/config/config/yaml```

