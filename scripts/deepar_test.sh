#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=3 python model_test.py  dataset=cstr 'save_dir=ckpt/cstr/deepar/deepar_model.k_size\=16\,train.max_epochs_stop\=20/2020-12-03_03-42-03' model=deepar model.k_size=16
CUDA_VISIBLE_DEVICES=3 python model_test.py  dataset=winding 'save_dir=ckpt/winding/deepar/deepar_model.k_size\=16\,train.max_epochs_stop\=20/2020-12-03_04-21-54' model=deepar model.k_size=16
CUDA_VISIBLE_DEVICES=3 python model_test.py  dataset=west 'save_dir=ckpt/west/deepar/deepar_model.k_size\=16\,train.max_epochs_stop\=20/2020-12-03_04-17-58' model=deepar model.k_size=16