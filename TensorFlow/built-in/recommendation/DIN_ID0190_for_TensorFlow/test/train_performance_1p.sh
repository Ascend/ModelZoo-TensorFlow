#!/bin/bash


#集合通信参数,不需要修改
export RANK_SIZE=1
export JOB_ID=10087

#export ASCEND_DEVICE_ID=
export OP_NO_REUSE_MEM=StridedSliceD
# 数据集路径,保持为空,不需要修改
data_path=""

#精度参数
#precision_mode="must_keep_origin_dtype"

#网络名称,同目录名称,需要模型审视修改
Network="DIN_ID0190_for_TensorFlow"

#训练batch_size,,需要模型审视修改
batch_size=1024

# 指定训练所使用的npu device卡id
#device_id=0
# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
#if [[ $data_path == "" ]];then
#    echo "[Error] para \"data_path\" must be confing"
#    exit 1
#fi
# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
#if [ $ASCEND_DEVICE_ID ];then
#    device_id=${ASCEND_DEVICE_ID}
#    echo "device id is ${ASCEND_DEVICE_ID}"
#elif [ ${device_id} ];then
#    export ASCEND_DEVICE_ID=${device_id}
#    echo "device id is ${ASCEND_DEVICE_ID}"
#else
#    "[Error] device id must be config"
#    exit 1
#fi
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

#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
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
#数据集处理
#ln -nsf ${data_path} $cur_path/data

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
cd $cur_path/examples
sed -i  "s|./data|$data_path|g" din_demo.py
sed -i  "s|epochs=5|epochs=2|g" din_demo.py


RANK_ID_START=0
RANK_SIZE=1

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
   echo "Device ID: $RANK_ID"
   export RANK_ID=$RANK_ID
   export ASCEND_DEVICE_ID=$RANK_ID
   ASCEND_DEVICE_ID=$RANK_ID
  if [   -d $cur_path/test/output/${ASCEND_DEVICE_ID} ];then
     rm -rf $cur_path/test/output/${ASCEND_DEVICE_ID}
     mkdir -p $cur_path/test/output/${ASCEND_DEVICE_ID}
  else
     mkdir -p $cur_path/test/output/${ASCEND_DEVICE_ID}
  fi

    corenum=`cat /proc/cpuinfo |grep 'processor' |wc -l`
    let a=RANK_ID*${corenum}/8
    let b=RANK_ID+1
    let c=b*${corenum}/8-1
    if [ "x${bind_core}" != x ];then
        bind_core="taskset -c $a-$c"
    fi

  nohup  taskset -c $a-$c python3.7  din_demo.py > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done

wait
sed -i 's///g' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log
sed -i 's//\n/g' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log
##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
train_time=`grep -rn "val_binary_crossentropy" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |awk -F ' ' '{print $5}'|awk -F 'ms' '{print $1}'|awk '{sum+=$1} END {print"",sum/NR}' |awk '$1=$1'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${train_time}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
#吞吐量
ActualFPS=${FPS}
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep "loss" ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F 'loss: ' '{print $2}'|awk -F ' ' '{print $1}' > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "CompileTime = ${CompileTime}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
