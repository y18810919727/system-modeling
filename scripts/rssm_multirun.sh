#!/usr/bin/env bash
cd ..

# Train
CUDA_VISIBLE_DEVICES=0 python model_train.py --multirun  dataset=nl save_dir=rssm model=rssm model.D=1,3,5,10 random_seed=0 dataset.history_length=1 dataset.forward_length=499 ct_time=true sp=0.25,0.5,0.75,1 #train.lr_reduce=false,true

# Test
#CUDA_VISIBLE_DEVICES=3 python model_test.py  dataset=west 'save_dir=ckpt/west/vrnn/vrnn_model.D\=3/2020-12-03_07-20-37' model=vrnn model.D=3
