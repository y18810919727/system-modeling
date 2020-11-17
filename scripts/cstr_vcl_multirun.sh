#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=2 python model_train.py --multirun model.D=1,5,15 dataset=cstr model.k_size=16 model.dynamic.num_linears=8
