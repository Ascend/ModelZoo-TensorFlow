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

#!/bin/bash

code_dir=$1
work_dir=$2
dataset_path=$3
output_path=$4

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
python3.7 ${code_dir}/main.py --data_path=${dataset_path} --output_path=${output_path} \
  --task_mode="train" \
  --master="/root/projects/meta-pseudo-labels-tf1-gpu/results/worker" \
  --dataset_name="cifar10_4000_mpl" \
  --model_type="wrn-28-2" \
  --optim_type="momentum" \
  --lr_decay_type="cosine" \
  --nouse_augment \
  --alsologtostderr \
  --running_local_dev \
  --image_size=32 \
  --num_classes=10 \
  --log_every=50 \
  --save_every=100 \
  --train_batch_size=64 \
  --eval_batch_size=64 \
  --uda_data=7 \
  --weight_decay=5e-4 \
  --num_train_steps=300000 \
  --augment_magnitude=16 \
  --batch_norm_batch_size=256 \
  --dense_dropout_rate=0.2 \
  --ema_decay=0.995 \
  --label_smoothing=0.15 \
  --mpl_teacher_lr=0.05 \
  --mpl_teacher_lr_warmup_steps=5000 \
  --mpl_student_lr=0.05 \
  --mpl_student_lr_wait_steps=1000 \
  --mpl_student_lr_warmup_steps=5000 \
  --uda_steps=5000 \
  --uda_temp=0.7 \
  --uda_threshold=0.6 \
  --uda_weight=8

if [ $? -eq 0 ];
then
    echo "[CANN-ZhongZhi] train return success"
else
    echo "[CANN-ZhongZhi] train return failed"
fi

######训练后把需要备份的内容保存到output_path######
###此处每个网络不同，视情况添加cp
cp -r ${work_dir} ${output_path}

######训练后输出目录文件确认######
echo "[CANN-ZhongZhi] after train - list my work files[${work_dir}]:"
ls -al ${work_dir}
echo ""

echo "[CANN-ZhongZhi] after train - list my output files[${output_path}]:"
ls -al ${output_path}
echo ""
