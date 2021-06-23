#!/usr/bin/env bash
cd ..

# Train
CUDA_VISIBLE_DEVICES=3 python model_train.py --multirun  dataset=southeast,west,cstr,winding model=informer train.max_epochs_stop=5 train.epochs=200 train.batch_size=512 save_dir=informer

# Test
#CUDA_VISIBLE_DEVICES=3 python model_test.py  dataset=west 'save_dir=ckpt/west/vrnn/vrnn_model.D\=3/2020-12-03_07-20-37' model=vrnn model.D=3
