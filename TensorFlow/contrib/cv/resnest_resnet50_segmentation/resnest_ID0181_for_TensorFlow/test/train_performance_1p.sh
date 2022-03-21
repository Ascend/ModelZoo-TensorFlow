#!/bin/bash
cur_path=`pwd`/../
################基础配置参数，需要模型审视修改#######################
# 必选字段(必须在此处定义的参数): Network batch_size resume RANK_SIZE
# 网络名称，同目录名称
Network="resnest_ID0181_for_TensorFlow"
# 训练batch_size
batch_size=10
# 训练使用的npu卡数
export RANK_SIZE=1
# export SOC_VERSION=Ascend910
# export HCCL_CONNECT_TIMEOUT=200
# export RANK_TABLE_FILE=./hccl_8p.json
# export RANK_INDEX=0
# export JOB_ID=10086
# export PRINT_MODEL=1
# 加载数据集进程数
workers=184
# 数据集路径,保持为空,不需要修改
data_path=""
ckpt_path=""
# 训练epoch
train_epochs=1
# 指定训练所使用的npu device卡id
# 学习率
learning_rate=0.0001

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --workers* ]];then
        workers=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

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


#################创建日志输出目录，不需要修改#################
cd $cur_path

#################启动训练脚本#################


wait

# 训练开始时间，不需要修改
start_time=$(date +%s)

#for((i=0;i<8;i++));
# for i in 0 1 2 3 4 5 6 7
# do
#export ASCEND_DEVICE_ID=$i
export RANK_ID=0
#ASCEND_DEVICE_ID=$i
#echo "ASCEND_DEVICE_ID:$ASCEND_DEVICE_ID"
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi
echo $ASCEND_DEVICE_ID
# sed -i "s#'./train_data/cityspaces_train.tfrecords'#'${data_path}/cityscapes_train.tfrecords'#g" npu_8_distribute_train.py
# sed -i "s#'./save_gpu_model/resnest.ckpt-46'#'${ckpt_path}/resnest.ckpt-46'#g" npu_8_distribute_train.py
echo ${ckpt_path}/resnest.ckpt-46
echo ${data_path}/cityspaces_train.tfrecords
python3 npu_8_distribute_train.py \
    --train_data=${data_path}/cityscapes_train.tfrecords \
    --pre_model=${ckpt_path}/resnest.ckpt-46  > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    #参数回改
    # sed -i "s#'${data_path}/cityspaces_train.tfrecords'#./train_data/cityscapes_train.tfrecords#g" npu_8_distribute_train.py
    # sed -i "s#'${ckpt_path}/resnest.ckpt-46'#./save_gpu_model/resnest.ckpt-46#g" npu_8_distribute_train.py
wait
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

##################获取训练数据################


# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
TrainingTime=`grep time  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk '{print $2}'|awk -F "," '{print $1}'`
# 打印，不需要修改
echo "Final Performance images/sec : $TrainingTime"

# 输出训练精度,需要模型审视修改
train_accuracy=`grep accuracy ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk '{print $13}'|awk -F "," '{print $1}'`
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

# 获取性能数据，不需要修改
# 吞吐量
#ActualFPS=${FPS}
# 单迭代训练时长
#TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${FPS}'/2}'`
ActualFPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${TrainingTime}'}'`
# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep loss ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $10}'|awk -F "," '{print $1}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "accuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log