#!/usr/bin/env bash
cd ..
#CUDA_VISIBLE_DEVICES=3 python model_test.py --multirun  dataset=cstr,winding,west save_dir=srnn model=srnn

#CUDA_VISIBLE_DEVICES=3 python model_test.py save_dir=ckpt/winding/srnn/srnn_/2020-11-25_19-48-03 model=srnn dataset=winding
#CUDA_VISIBLE_DEVICES=3 python model_test.py save_dir=ckpt/cstr/srnn/srnn_/2020-11-25_14-57-52 model=srnn dataset=cstr
CUDA_VISIBLE_DEVICES=3 python model_test.py save_dir=ckpt/west/srnn/srnn_/2020-11-25_21-46-40 model=srnn dataset=west test.plt_cnt=40
