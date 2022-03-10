#!/usr/bin/env bash
cd ..

# Train
CUDA_VISIBLE_DEVICES=0 python model_train.py --multirun  dataset=southeast save_dir=rssm model=rssm model.D=1,3

# Test
#CUDA_VISIBLE_DEVICES=3 python model_test.py  dataset=west 'save_dir=ckpt/west/vrnn/vrnn_model.D\=3/2020-12-03_07-20-37' model=vrnn model.D=3
