#!/usr/bin/env bash
cd ..
#CUDA_VISIBLE_DEVICES=3 python model_test.py --version cl --dataset west-part --L 1 --use_cuda --save_dir test_cl --eval_epochs 10 --reset
#CUDA_VISIBLE_DEVICES=3 python model_test.py --version cl --dataset west-part --use_cuda --save_dir test_cl_h16
#CUDA_VISIBLE_DEVICES=3 python model_test.py --version cl --dataset west-part --L 5 --use_cuda --save_dir test_cl_h16 --eval_epochs 10 --reset --epochs 400 --length 200 --state_size 16

#CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-05/07-22-05/0 dataset=cstr1 model.k_size=16
#CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-05/07-22-05/1 dataset=cstr2 model.k_size=16
#CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-12/03-08-48/0 dataset=cstr model.k_size=16 model.dynamic.num_linears=8 model.D=1
#CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-12/03-08-48/1 dataset=cstr model.k_size=16 model.dynamic.num_linears=8 model.D=5
#CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-12/03-08-48/2 dataset=cstr model.k_size=16 model.dynamic.num_linears=8 model.D=15

#CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-15/12-44-59/0 dataset=west model.k_size=16 model.dynamic.num_linears=8 model.D=1
#CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-15/12-44-59/1 dataset=west model.k_size=16 model.dynamic.num_linears=8 model.D=5
#CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-15/12-44-59/2 dataset=west model.k_size=16 model.dynamic.num_linears=8 model.D=15


#CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-16/02-34-23/0 dataset=west model.k_size=16 model.dynamic.num_linears=8 model.D=10
#CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-16/02-34-23/1 dataset=west model.k_size=16 model.dynamic.num_linears=8 model.D=20


#CUDA_VISIBLE_DEVICES=3 python model_train.py --multirun model.D=10,20 dataset=west model.k_size=16 model.dynamic.num_linears=8

CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-16/13-45-39/0 dataset=winding model.k_size=16 model.dynamic.num_linears=8 model.D=1
CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-16/13-45-39/1 dataset=winding model.k_size=16 model.dynamic.num_linears=8 model.D=5
CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-16/13-45-39/2 dataset=winding model.k_size=16 model.dynamic.num_linears=8 model.D=15
CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-16/13-45-39/3 dataset=winding model.k_size=16 model.dynamic.num_linears=8 model.D=20
CUDA_VISIBLE_DEVICES=3 python model_test.py +save_dir=multirun/2020-11-16/13-45-39/4 dataset=winding model.k_size=16 model.dynamic.num_linears=8 model.D=25
