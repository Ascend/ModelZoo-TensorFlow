#!/bin/bash

rm -rf /home/DeepMatchVo_ID2363_for_TensorFlow/ckpt-yyw/

python3 train.py --dataset_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/kitti/genertate_wty'  --checkpoint_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/ckpt-yyw' --max_steps 300000 --save_freq 3000 --learning_rate 0.001 --num_scales 1  --continue_train=False --match_num=100 
