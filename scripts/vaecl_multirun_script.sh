#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=2 python model_train.py --multirun model.D=1,5,10,15,20 dataset=cstr,winding,west save_dir=vaecl_net_decoder model=vaecl
