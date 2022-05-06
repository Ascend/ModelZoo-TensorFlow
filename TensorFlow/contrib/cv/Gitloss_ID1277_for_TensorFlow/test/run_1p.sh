#/bin/bash

# source activate /home/ma-user/miniconda3/envs/TensorFlow-1.15.0
# pip install tflearn

cd ../

python gitloss.py --update_centers=1000 --lambda_c=1.0 --lambda_g=1.0 --steps=8000