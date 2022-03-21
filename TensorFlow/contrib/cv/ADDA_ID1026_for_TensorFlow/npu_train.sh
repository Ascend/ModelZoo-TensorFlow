"""
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

#!/bin/bash

code_dir=$1
work_dir=$2
dataset_path=$3
output_path=$4

pip install tflearn
pip install colorlog

#############训练前输入目录文件确认#########################
echo "[CANN-ZhongZhi] before train - list my run files[/usr/local/Ascend/ascend-toolkit]:"
ls -al /usr/local/Ascend/ascend-toolkit
echo ""

echo "[CANN-ZhongZhi] before train - list my code files[${code_dir}]:"
ls -al ${code_dir}
echo ""

echo "[CANN-ZhongZhi] before train - list my work files[${work_dir}]:"
ls -al ${work_dir}
echo ""

echo "[CANN-ZhongZhi] before train - list my dataset files[${dataset_path}]:"
ls -al ${dataset_path}
echo ""

echo "[CANN-ZhongZhi] before train - list my output files[${output_path}]:"
ls -al ${output_path}
echo ""

######环境变量修改######
###如果需要修改环境变量的，在此处修改
#设置日志级别为info
#export ASCEND_GLOBAL_LOG_LEVEL=1
#设置日志打屏到屏幕
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export TF_CPP_MIN_LOG_LEVEL=0
env > ${output_path}/my_env.log

######训练执行######
###此处每个网络执行命令不同，需要修改
#python3.7 ${code_dir}/LeNet.py --data_path=${dataset_path} --output_path=${output_path} --steps=100
# abort entire script on error
set -e


###111111111111111111111111111111111111111
## train base model on svhn
python ${code_dir}/tools/train_xnhg.py svhn train lenet lenet_svhn \
       --iterations 20 \
       --batch_size 128 \
       --display 10 \
       --lr 0.001 \
       --snapshot 1000 \
       --solver adam \
       --seed 0

echo 'Source only baseline:'
python ${code_dir}/tools/eval_classification.py mnist train lenet snapshot/lenet_svhn


# run adda svhn->mnist
python ${code_dir}/tools/train_adda_xnhg.py svhn:train mnist:train lenet adda_lenet_svhn_mnist \
       --iterations 30 \
       --batch_size 128 \
       --display 10 \
       --lr 0.0002 \
       --snapshot 1000 \
       --weights snapshot/lenet_svhn \
       --adversary_relu \
       --solver adam \
       --seed 0

python ${code_dir}/tools/eval_classification.py mnist train lenet snapshot/adda_lenet_svhn_mnist

# 222222222222222222222222222222222222222
# train base model on usps1800
python ${code_dir}/tools/train_xnhg.py usps1800 train lenet lenet_usps \
       --iterations 100 \
       --batch_size 128 \
       --display 2 \
       --lr 0.002 \
       --snapshot 100 \
       --solver adam\
       --seed 0

python ${code_dir}/tools/eval_classification.py mnist2000 train lenet snapshot/lenet_usps
## run adda usps1800->mnist2000
python ${code_dir}/tools/train_adda_xnhg.py usps1800:train mnist2000:train lenet adda_lenet_usps_mnist \
       --iterations 100 \
       --batch_size 128 \
       --display 2 \
       --lr 0.0002 \
       --snapshot 100 \
       --weights snapshot/lenet_usps \
       --adversary_relu \
       --solver adam \
       --seed 0

#evaluate trained models
echo 'ADDA':
python ${code_dir}/tools/eval_classification.py mnist2000 train lenet snapshot/adda_lenet_usps_mnist



#3333333333333333333333333333
python ${code_dir}/tools/train_xnhg.py mnist2000 train lenet lenet_mnist \
       --iterations 300 \
       --batch_size 128 \
       --display 5 \
       --lr 0.0001 \
       --snapshot 100 \
       --solver adam \
       --seed 0

python ${code_dir}/tools/eval_classification.py usps1800 train lenet snapshot/lenet_mnist

python ${code_dir}/tools/train_adda_xnhg.py mnist2000:train usps1800:train lenet adda_lenet_mnist_usps \
       --iterations 300 \
       --batch_size 128 \
       --display 5 \
       --lr 0.0001 \
       --snapshot 100 \
       --weights snapshot/lenet_mnist \
       --adversary_relu \
       --solver adam \
       --seed 0

python ${code_dir}/tools/eval_classification.py usps1800 train lenet snapshot/adda_lenet_mnist_usps


if [ $? -eq 0 ];
then
    echo "[CANN-ZhongZhi] train return success"
else
    echo "[CANN-ZhongZhi] train return failed"
fi

######训练后把需要备份的内容保存到output_path######
###此处每个网络不同，视情况添加cp
cp -r ${work_dir} ${output_path}
#cp -r ${work_dir}/snapshot  ${output_path}
#cp -r ${work_dir}/profiling  ${output_path}

cp -r '/cache/profiling'  ${output_path}
######训练后输出目录文件确认######
echo "[CANN-ZhongZhi] after train - list my work files[${work_dir}]:"
ls -al ${work_dir}
echo ""

echo "[CANN-ZhongZhi] after train - list my output files[${output_path}]:"
ls -al ${output_path}
echo ""
