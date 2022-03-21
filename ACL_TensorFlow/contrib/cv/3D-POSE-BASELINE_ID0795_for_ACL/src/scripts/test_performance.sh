#!/bin/bash

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

# boot sh-file

export TF_CPP_MIN_LOG_LEVEL=2        # Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0 # Print log on terminal on(1), off(0)

# 我的修改
code_dir=${1}
data_dir=${2}
result_dir=${3}
obs_url=${4}

#当前路径,不需要修改
cur_path=$(pwd)

#集合通信参数,不需要修改
export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="3D-POSE-BASELINE_ID0795_for_TensorFlow"
#训练epoch
train_epochs=200
#训练batch_size
batch_size=64
#训练step
#train_steps=
#学习率
learning_rate=1e-3

pip install cdflib

##维测参数，precision_mode需要模型审视修改
#precision_mode="allow_mix_precision"
##维持参数，以下不需要修改
#over_dump=False
#data_dump_flag=False
#data_dump_step="10"
#profiling=False

## 帮助信息，不需要修改
#if [[ $1 == --help || $1 == -h ]];then
#    echo"usage:./train_performance_1P.sh <args>"
#    echo " "
#    echo "parameter explain:
#    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
#    --over_dump		           if or not over detection, default is False
#    --data_dump_flag		     data dump flag, default is False
#    --data_dump_step		     data dump step, default is 10
#    --profiling		           if or not profiling for performance debug, default is False
#    --data_dir		           source data of training
#    -h/--help		             show help message
#    "
#    exit 1
#fi
#
##参数校验，不需要修改
#for para in $*
#do
#    if [[ $para == --precision_mode* ]];then
#        precision_mode=`echo ${para#*=}`
#    elif [[ $para == --over_dump* ]];then
#        over_dump=`echo ${para#*=}`
#        over_dump_path=${cur_path}/output/overflow_dump
#        mkdir -p ${over_dump_path}
#    elif [[ $para == --data_dump_flag* ]];then
#        data_dump_flag=`echo ${para#*=}`
#        data_dump_path=${cur_path}/output/data_dump
#        mkdir -p ${data_dump_path}
#    elif [[ $para == --data_dump_step* ]];then
#        data_dump_step=`echo ${para#*=}`
#    elif [[ $para == --profiling* ]];then
#        profiling=`echo ${para#*=}`
#        profiling_dump_path=${cur_path}/output/profiling
#        mkdir -p ${profiling_dump_path}
#    elif [[ $para == --data_dir* ]];then
#        data_dir=`echo ${para#*=}`
#    fi
#done

#校验是否传入data_dir,不需要修改
if [[ $data_dir == "" ]]; then
  echo "[Error] para \"data_dir\" must be confing"
  exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../
for ((RANK_ID = $RANK_ID_START; RANK_ID < $((RANK_SIZE + RANK_ID_START)); RANK_ID++)); do
  #设置环境变量，不需要修改
  echo "Device ID: $ASCEND_DEVICE_ID"
  export RANK_ID=$RANK_ID
  #创建DeviceID输出目录，不需要修改
  if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ]; then
    rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
  else
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
  fi

  #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
  #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
  nohup python3 ${code_dir}/predict_3dpose.py \
    --data_dir ${data_dir}"/h36m" \
    --result ${result_dir} \
    --obs_dir ${obs_url} \
    --cameras_path ${data_dir}"/h36m/metadata.xml" \
    --camera_frame \
    --residual \
    --batch_norm \
    --dropout 0.5 \
    --max_norm \
    --evaluateActionWise \
    --epochs $train_epochs \
    --sample \
    --load 4874200
  #              --learning_rate $learning_rate \
  #		          --batch_size $batch_size > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

##训练结束时间，不需要修改
#end_time=$(date +%s)
#e2e_time=$(($end_time - $start_time))
##结果打印，不需要修改
#echo "------------------ Final result ------------------"
##输出性能FPS，需要模型审视修改
#cost_time=$(grep 'Working on epoch' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk '{print $14}' | tail -2 | head -n 1)
##avg_cost=`awk 'BEGIN{printf "%f\n",'${cost_sum}'/'${cost_count}'}'`
#FPS=$(awk 'BEGIN{printf "%d\n",'${batch_size}'/'${cost_time}'*1000}')
##FPS=`grep 'done in' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $6}'`
##打印，不需要修改
#echo "Final Performance images/sec : $FPS"
##输出训练精度,需要模型审视修改
#train_accuracy=$(grep Average $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk '{print $2}')
##打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
#echo "E2E Training Duration sec : $e2e_time"
##性能看护结果汇总
##训练用例信息，不需要修改
#BatchSize=${batch_size}
#DeviceType=$(uname -m)
#CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
###获取性能数据，不需要修改
##吞吐量
#ActualFPS=${FPS}
##单迭代训练时长
#TrainingTime=$(awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}')
##从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
#grep 'Working on epoch' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $10}' >$cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
##最后一个迭代loss值，不需要修改
#ActualLoss=$(awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt)
##关键信息打印到${CaseName}.log中，不需要修改
#echo "Network = ${Network}" >$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "RankSize = ${RANK_SIZE}" >>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "BatchSize = ${BatchSize}" >>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "DeviceType = ${DeviceType}" >>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "CaseName = ${CaseName}" >>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "ActualFPS = ${ActualFPS}" >>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "TrainingTime = ${TrainingTime}" >>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "TrainAccuracy = ${train_accuracy}" >>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "ActualLoss = ${ActualLoss}" >>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "E2ETrainingTime = ${e2e_time}" >>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
