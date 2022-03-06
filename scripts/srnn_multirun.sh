#!/usr/bin/env bash
cd ..
# Train
CUDA_VISIBLE_DEVICES=0 python model_train.py --multirun  dataset=ib save_dir=srnn_overshooting model=srnn model.D=1,3,5

# Test a sample
#CUDA_VISIBLE_DEVICES=3 python model_test.py  dataset=west 'save_dir=ckpt/west/srnn/srnn_model.D\=1/2020-12-03_02-48-55/' model=srnn model.D=1

