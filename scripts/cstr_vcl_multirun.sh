#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=2 python model_train.py --multirun model.D=1,5,10,15 dataset=cstr model.dynamic.num_linears=4,8 model.k_size=8,16 save_dir=vcl
CUDA_VISIBLE_DEVICES=2 python model_test.py --multirun model.D=1,5,10,15 dataset=cstr model.dynamic.num_linears=4,8 model.k_size=8,16 save_dir=vcl

# model.k_size=16
