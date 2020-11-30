#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=3 python model_train.py --multirun  dataset=cstr,winding,west save_dir=srnn model=srnn
