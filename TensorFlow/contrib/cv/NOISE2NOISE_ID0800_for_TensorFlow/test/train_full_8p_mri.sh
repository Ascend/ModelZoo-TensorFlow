#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="NOISE2NOISE_ID800_for_TensorFlow_Mri"
# 训练batch_size
batch_size=16

epoch=300

is_distributed='True'
is_loss_scale='True'
hcom_parallel='True'

graph_run_mode=1 # train
op_select_implmode='high_precision'
precision_mode='allow_mix_precision'

cur_path=`pwd`

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_TABLE_FILE=${cur_path}/../configs/8p.json
export JOB_ID=10087
export RANK_SIZE=8
RANK_ID_START=0
device_group="0 1 2 3 4 5 6 7"

#参数校验，不需要修改
for para in "$@"
do
    if [[ $para == --epoch* ]];then
        epoch=${para#*=}
    elif [[ $para == --is-distributed* ]];then
        is_distributed=${para#*=}
    elif [[ $para == --is-loss-scale* ]];then
        is_loss_scale=${para#*=}
    elif [[ $para == --graph-run-mode* ]];then
        graph_run_mode=${para#*=}
    elif [[ $para == --op-select-implmode* ]];then
        op_select_implmode=${para#*=}
    elif [[ $para == --precision-mode* ]];then
        precision_mode=${para#*=}
    elif [[ $para == --hcom-parallel* ]];then
        hcom_parallel=${para#*=}
    fi
done

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#训练开始时间，不需要修改
start_time=$(date +%s)


RANK_ID=$RANK_ID_START
for device_id  in ${device_group};
do

  if  [ x"${device_id}" = x ] ;
  then
      echo "turing train fail" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log
      exit
  else
      export DEVICE_ID=${device_id}
  fi

  #设置环境变量，不需要修改
  export RANK_ID=$RANK_ID
  echo "Device ID: $DEVICE_ID"
  echo "RANK ID: $RANK_ID"
  export ASCEND_DEVICE_ID=$DEVICE_ID
  ASCEND_DEVICE_ID=$DEVICE_ID


  #################创建日志输出目录，不需要修改#################
  if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
      rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
      mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
  else
      mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
  fi

  nohup python3.7 config_mri.py \
          --epoch="${epoch}" \
          --is-distributed="${is_distributed}" \
          --is-loss-scale="${is_loss_scale}" \
          --hcom-parallel="${hcom_parallel}" \
          --precision-mode="${precision_mode}" \
          --op-select-implmode="${op_select_implmode}" \
          --graph-run-mode="${graph_run_mode}" > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

  RANK_ID=$((RANK_ID + 1))

  sleep 1

done
wait


##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

ASCEND_DEVICE_ID=0

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能，需要模型审视修改
sec_per_iter=`grep 'sec/iter=' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{if (NR>=3){sum+=$(12)}} END {sum/=(NR-2);print sum}'`
FPS=`echo "${batch_size} ${sec_per_iter} ${RANK_SIZE}"|awk '{print ($1 / $2 * $3)}'`
FPS=${FPS%,*}
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
train_accuracy=`grep 'test_db_clamped=' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $8}'`
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=${sec_per_iter}

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'train_loss=' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $(6)}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log