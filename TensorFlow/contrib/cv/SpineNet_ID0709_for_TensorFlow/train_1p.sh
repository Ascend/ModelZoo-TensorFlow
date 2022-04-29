#!/usr/bin/env python3
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
echo "`date +%Y%m%d%H%M%S`-[INFO] start to run train_1p.sh "

#####################
# 【必填】对标性能和精度指标，来源于论文或者GPU复现
benchmark_fps=""
benchmark_accu=""
#####################
# 训练标准化参数列表，请通过参数进行相关路径传递
cur_dir=`pwd`
# 1、数据集路径地址，若不涉及则跳过
data_dir=`pwd`
# 2、代码地址，若不涉及则跳过
code_dir=`pwd`
# 3、训练输出的地址，若不涉及则跳过
result_dir=""
for para in $*
do
    if [[ $para == --data_dir* ]];then
        data_dir=`echo ${para#*=}`
    fi
done

export SPINENETHOME=$code_dir
export PYTHONPATH=$SPINENETHOME:$SPINENETHOME/official:$SPINENETHOME/official/efficientnet:$SPINENETHOME/official/detection:$PYTHONPATH
#####################
# 训练执行拉起命令，打屏信息输出到train_output.log文件
cd $cur_path/
python ${code_dir}/official/detection/main.py \
 --model_dir="${result_dir}" --use_tpu=False --mode=train --eval_after_training=True \
 --config_file="${code_dir}/official/detection/configs/spinenet/spinenet49S_retinanet.yaml" \
 --params_override="{ train: { train_batch_size : 4, train_file_pattern: ${data_dir}/train-* }, eval: { val_json_file: ${data_dir}/annotations/instances_val2017.json, eval_file_pattern: ${data_dir}/val-* } }"
 
 
echo "`date +%Y%m%d%H%M%S`-[INFO] finish to run train_1p.sh "