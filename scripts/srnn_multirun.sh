#!/usr/bin/env bash
cd ..
# Train
CUDA_VISIBLE_DEVICES=3 python model_train.py --multirun  dataset=west save_dir=srnn model=srnn model.D=1,3,5,10 ct_time=true sp=0.25,0.5,0.75,1

# Test a sample
#CUDA_VISIBLE_DEVICES=3 python model_test.py  dataset=west 'save_dir=ckpt/west/srnn/srnn_model.D\=1/2020-12-03_02-48-55/' model=srnn model.D=1

