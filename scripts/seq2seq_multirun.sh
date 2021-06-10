#!/usr/bin/env bash
cd ..

# Train
CUDA_VISIBLE_DEVICES=3 python model_train.py --multirun  dataset=cstr,winding,west save_dir=seq2seq model=seq2seq train.max_epochs_stop=30 train.epochs=200 train.batch_size=128

# Test
#CUDA_VISIBLE_DEVICES=3 python model_test.py  dataset=west 'save_dir=ckpt/west/vrnn/vrnn_model.D\=3/2020-12-03_07-20-37' model=vrnn model.D=3
