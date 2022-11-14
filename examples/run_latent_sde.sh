#!/usr/bin/env bash
# 跑之前确认一下:
# 1. save_dir
# 2. 显卡编号
# 3. 多参数网格搜索需要加 --multirun
# 4. 新加的那几个参数是否在debug 中验证。

# Tips: screen -L -t log -S {session name} {xxx.sh} : screen日志会存储在当前目录 下的 screenlog.0中
cd ../
# Train

CUDA_VISIBLE_DEVICES=3 python model_train.py dataset=cstr save_dir=tmp model=latent_sde model.inter=zero train.init_weights=False ct_time=true sp=0.5 random_seed=0 train.batch_size=256 train.epochs=1 train.eval_epochs=1
