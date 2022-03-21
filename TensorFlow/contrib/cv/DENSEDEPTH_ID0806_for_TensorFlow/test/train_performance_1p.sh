#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="DENSEDEPTH_ID0806_for_TensorFlow"
# 训练batch_size
batch_size=4

epochs=3
steps=50
#test_data="./dataset/nyu_test.zip"
#train_tfrecords="./dataset/nyu_data.tfrecords"
data_path=''
is_distributed='False'
is_loss_scale='True'
hcom_parallel='False'

op_select_implmode='high_precision'
precision_mode='allow_mix_precision'

# 训练使用的npu卡数
export RANK_SIZE=1

# 指定训练所使用的npu device卡id
#device_id=0

#参数校验，不需要修改
for para in "$@"
do
    if [[ $para == --epochs* ]];then
        epochs=${para#*=}
    elif [[ $para == --bs* ]];then
        batch_size=${para#*=}
    elif [[ $para == --data_path* ]];then
        data_path=${para#*=}
    elif [[ $para == --is_distributed* ]];then
        is_distributed=${para#*=}
    elif [[ $para == --is_loss_scale* ]];then
        is_loss_scale=${para#*=}
    elif [[ $para == --op_select_implmode* ]];then
        op_select_implmode=${para#*=}
    elif [[ $para == --precision_mode* ]];then
        precision_mode=${para#*=}
    elif [[ $para == --hcom_parallel* ]];then
        hcom_parallel=${para#*=}
    elif [[ $para == --steps* ]];then
        steps=${para#*=}
    fi
done

# # 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
# if [ $ASCEND_DEVICE_ID ];then
#     echo "device id is ${ASCEND_DEVICE_ID}"
# elif [ ${device_id} ];then
#     export ASCEND_DEVICE_ID=${device_id}
#     echo "device id is ${ASCEND_DEVICE_ID}"
# else
#     "[Error] device id must be config"
#     exit 1
# fi


###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi


#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi
cp ${data_path}/dataset/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5  ${cur_path}
nohup python3.7  train.py \
          --test_data="${data_path}/dataset/nyu_test.zip" \
          --bs="${batch_size}" \
          --train_tfrecords="${data_path}/dataset/nyu_data.tfrecords" \
          --epochs="${epochs}" \
          --steps="${steps}" \
          --full \
          --op_select_implmode="${op_select_implmode}" \
          --precision_mode="${precision_mode}" \
          --hcom_parallel="${hcom_parallel}" \
          --is_distributed="${is_distributed}" \
          --is_loss_scale="${is_loss_scale}" > "${test_path_dir}"/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait


##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
iters_per_epoch=`grep 'iters/epoch:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $(5)}'`
#avre_time=`grep 'loss:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F '[ s]' '{if (NR>=3){sum+=$(3)}} END {sum/=(NR-2);print sum}'`
avre_time=`grep 'loss:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $3}'|awk -F 's' '{print $1}'`
FPS=`echo "${batch_size} ${iters_per_epoch} ${avre_time}"|awk '{print ($(1) * $(2) / $(3))}'`
FPS=${FPS%,*}
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
# 用指标rms代替，输出最小的rms值
train_accuracy='None'
#train_accuracy=`grep 'rms' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'BEGIN {rms = 1} {if($(10) < rms) rms = $(10)} END {print rms}'`
# 打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`echo "${batch_size} ${FPS}"|awk '{print ($1/$2)*1000}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'loss:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $(6)}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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