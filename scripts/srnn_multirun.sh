#!/usr/bin/env bash
cd ..
# Train
CUDA_VISIBLE_DEVICES=1 python model_train.py --multirun  dataset=cstr,winding,west save_dir=srnn_overshooting model=srnn model.D=5,10

# Test a sample
CUDA_VISIBLE_DEVICES=3 python model_test.py  dataset=west 'save_dir=ckpt/west/srnn/srnn_model.D\=1/2020-12-03_02-48-55/' model=srnn model.D=1

