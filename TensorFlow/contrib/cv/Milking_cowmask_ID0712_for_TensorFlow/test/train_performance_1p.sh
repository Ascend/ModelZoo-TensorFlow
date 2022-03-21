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

# !/bin/bash
cur_path=`pwd`/.. #代码路径
export RANK_SIZE=1
export JOB_ID=10087
#export ASCEND_DEVICE_ID=0

#模型训练参数
Network="Milking_cowmask_ID0712_for_TensorFlow"
data_path=''
batch_size=1
num_epochs=1

#pip install -r ${cur_path}/requirements.txt
# 帮助信息，
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --Network               name of the network will be trained
    --data_path             source data of training , default is ${cur_path}/dataset/
    --result_dir            output path, default is ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt/
    --batch_size            batchsize of input per step, default is 256
    --num_epochs            num of epochs, default is 1
    -h/--help               show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --num_epochs* ]];then
        num_epochs=`echo ${para#*=}`
    fi
    if [[ $para == --conda_name* ]];then
      conda_name=`echo ${para#*=}`
	  source set_conda.sh
	  source activate $conda_name
   fi
done

#检查data_path
if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path \" must be config"
   exit 1
fi

#训练过程
cd ${cur_path}

start=$(date +%s)
#ll /npu/traindata/ID0712_Milking_cowmask_for_TensorFlow/dataset
du -sh /npu/traindata/ID0712_Milking_cowmask_for_TensorFlow/dataset

if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
  else
    mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
fi
echo "**************************"
python3 train.py \
  --dataset_path=${data_path}/dataset \
  --model_path=${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt \
  --num_epochs=${num_epochs} \
  --batch_size=${batch_size} > ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1
wait

end=$(date +%s)
e2e_time=$(( $end - $start ))
echo "Final Training Duration sec : $e2e_time"


#结果打印
echo "------------------ Final result ------------------"
#输出性能
TrainingTime=`grep "perf:" ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $6}'`
FPS=`grep "FPS:" ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $8}'`
accuracy=`grep "accuracy:" ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $7}'`
##打印
echo "Final Performance TrainingTime : $TrainingTime"
echo "Final Performance images/sec : $FPS"
echo "Final Accuracy : ${accuracy}"

BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

#输出train_loss
grep "FPS:" ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $10}' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
# 退出anaconda环境
if [ -n "$conda_name"];then
   conda deactivate
fi
