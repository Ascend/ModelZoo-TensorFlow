#/bin/bash

# /home/ma-user/work/npu-v4

# source activate /home/ma-user/miniconda3/envs/TensorFlow-1.15.0

cd ../

python3 main.py --path='./data/mIN.pkl' --data_set='mIN' --num_shots=1 --num_epoch=6 --num_ways=5