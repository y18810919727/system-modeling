#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=3 python model_train.py --multirun model.D=1,5,15,20,25 dataset=winding model.k_size=16 model.dynamic.num_linears=8
