#!/bin/sh
### Modelarts Platform train command

export TF_CPP_MIN_LOG_LEVEL=2               ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0        ## Print log on terminal on(1), off(0)
export ASCEND_GLOBAL_LOG_LEVEL=3
code_dir=${1}
data_dir=${2}
result_dir=${3}
pip install mock &&
pip install -r ${code_dir}/pip-requirements.txt &&
#start exec
#eval_after_train
python ${code_dir}/compare_gan/main.py  \
  --model_dir=${result_dir} \
	--tfds_data_dir=${data_dir} \
	--gin_config=${code_dir}/compare_gan/resnet_cifar10_false.gin \
	--myevalinception=${data_dir}/frozen_inception_v1_2015_12_05.tar.gz \
	--schedule=eval_after_train \
	--eval_every_steps=60000 \



