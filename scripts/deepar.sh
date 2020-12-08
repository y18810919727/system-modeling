#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=0 python model_train.py --multirun dataset=cstr,west,winding model.k_size=16 model=deepar train.max_epochs_stop=20 save_dir=deepar
