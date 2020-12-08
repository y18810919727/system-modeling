#!/usr/bin/env bash
cd ../..

# Train
CUDA_VISIBLE_DEVICES=1 python model_train.py --multirun  dataset=cstr,winding,west save_dir=vaecl model=vaecl model.D=1,3,5


# Test a sample
CUDA_VISIBLE_DEVICES=3 python model_test.py  dataset=west 'save_dir=ckpt/west/vaecl/vaecl_model.D\=3/2020-12-01_13-44-08/' model=vaecl model.D=3
