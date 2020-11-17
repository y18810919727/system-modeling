#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=3 python model_train.py --multirun model.D=1,5,10,15,20 dataset=west model.k_size=16 model.dynamic.num_linears=8
