# SE-VAE

## 模型训练

- multirun：根据参数的笛卡尔积同时训练多个模型，
``` python
CUDA_VISIBLE_DEVICES=3 python model_train.py --multirun model.D=1,5,10,15,20 dataset=west model.k_size=16 model.dynamic.num_linears=8
```
 使用3号GPU，参数D分别为1,5,10,15,20五种情况,训练五个模型，其他参数默认。训练模型与日志存储在```multirun/${now:%Y-%m-%d_%H-%M-%S}```
 
> 参考```scripts/west_vcl_script.sh```文件
 
 - 单模型训练：不加multirun参数
 
``` python
CUDA_VISIBLE_DEVICES=3 python model_train.py train.batch_size=32 dataset=winding model.D=1
```
训练模型与日志存储在```ckpt/${now:%Y-%m-%d_%H-%M-%S}```中
 
## 模型测试
运行```model_test``` 测试模型
``` python
CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-16/13-45-39/4 dataset=winding model.k_size=16 model.dynamic.num_linears=8 model.D=25
```
自动检测```best.pth```模型，其他参数需要与训练时的参数保持一致(包括模型版本、网络节点数等等)，测试之后会在目录下生成```figs```文件夹以及```test.out```日志文件
> 参考```scripts/test_vcl_script.sh```文件

## 部分参数介绍：
- ```dataset```: ```cstr``` 或 ```west```或，设定为```west```时自动从aliyun的oss下载数据文件,如果使用仿真数据测试，如果设定为```fake```，则采用仿真数据，需要先运行```data/fake_data_generator.py```生成仿真数据，
- ```D``` overshooting最大距离
- ```k_size``` latent state的大小
- ```version``` 选择算法模型，
    > 目前包含的版本如下:
    
    | 版本名      |   模型   |   备注   |
    | :-------- | --------:| :------: |
    | vaecl    |  VAE combinational linears  |   动态系统为自适应线性组合，线性解器 |
- ```plt_cnt```: 决定运行```model_test.py```时会画几个图在figs中
## hydra  
参考资料: https://hydra.cc/
