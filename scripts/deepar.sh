#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=1 python model_train.py --multirun dataset=west model=deepar save_dir=deepar random_seed=0 ct_time=true sp=0.25,0.5,0.75,1
