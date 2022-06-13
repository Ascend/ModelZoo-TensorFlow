#!/bin/bash

cur_path=`pwd`/../

export RANK_SIZE=16
export JOB_ID=10087
export ASCEND_GLOBAL_LOG_LEVEL_ETP_ETP_ETP=3

#基础参数，需要模型审视修改
#Batch Size
batch_size=1024
#网络名称，同目录名称
Network="MNIST_ID2481_for_TensorFlow2.X"
#训练epoch，可选
train_epochs=10
#参数配置
data_path="/npu/traindata/ID2481_CarPeting_TF2.X_MNIST"

############维测参数##############
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
if [[ $over_dump == True ]];then
    over_dump_path=$cur_path/test/overflow_dump #此处cur_path为代码根目录
    mkdir -p ${over_dump_path}
fi
data_dump_flag=False
data_dump_step="10"
profiling=False
use_mixlist=False
mixlist_file="ops_info.json"
fusion_off_flag=False
fusion_off_file="fusion_switch.cfg"
############维测参数##############

############维测参数##############
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --use_mixlist* ]];then
        use_mixlist=`echo ${para#*=}`
    elif [[ $para == --mixlist_file* ]];then
        mixlist_file=`echo ${para#*=}`
    elif [[ $para == --fusion_off_flag* ]];then
        fusion_off_flag=`echo ${para#*=}`
    elif [[ $para == --fusion_off_file* ]];then
        fusion_off_file=`echo ${para#*=}`
	elif [[ $para == --server_index* ]];then
        server_index=`echo ${para#*=}`
    elif [[ $para == --conf_path* ]];then
        conf_path=`echo ${para#*=}`
    fi
done
############维测参数##############

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path\" must be config"
   exit 1
fi

export RANK_SIZE=16
export JOB_ID=10087
rank_size=8
#export RANK_TABLE_FILE=$cur_path/../scripts/rank_table_16p.json
nohup python3 $cur_path/test/set_ranktable.py --npu_nums=$((RANK_SIZE/rank_size)) --conf_path=$conf_path
export RANK_TABLE_FILE=${cur_path}/test/rank_table.json
export HCCL_CONNECT_TIMEOUT=600
export RANK_INDEX=0
RANK_ID_START=0
RANK_SIZE=16


cd $cur_path
start=$(date +%s)
##############执行训练##########
RANK_ID_START=0
for((RANK_ID=$((rank_size*server_index));RANK_ID<$((((server_index+1))*rank_size));RANK_ID++));
do
    # 设置环境变量
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=`expr ${RANK_ID} - $((rank_size*server_index))`
    ASCEND_DEVICE_ID=`expr ${RANK_ID} - $((rank_size*server_index))`
    export DEVICE_ID=${ASCEND_DEVICE_ID}
    echo "DEVICE ID: $ASCEND_DEVICE_ID"

    # 创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID
    else
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID
    fi
    cpucount=`lscpu | grep "CPU(s):" | head -n 1 | awk '{print $2}'`
    cpustep=`expr $cpucount / 8`
    echo "taskset c steps:" $cpustep
    let a=ASCEND_DEVICE_ID*$cpustep
    let b=ASCEND_DEVICE_ID+1
    let c=b*$cpustep-1

    nohup taskset -c $a-$c  python3 mnist_main.py --precision_mode=${precision_mode} \
             --over_dump=${over_dump} \
             --over_dump_path=${over_dump_path} \
             --data_dump_flag=${data_dump_flag} \
             --data_dump_step=${data_dump_step} \
             --data_dump_path=${data_dump_path} \
             --profiling=${profiling} \
             --use_mixlist=${use_mixlist} \
             --fusion_off_flag=${fusion_off_flag} \
             --mixlist_file=${mixlist_file} \
             --fusion_off_file=${fusion_off_file} \
             --profiling_dump_path=${profiling_dump_path} \
             --model_dir=./ckpt \
             --train_epochs=10 \
             --distribution_strategy=one_device \
             --num_gpus=1 \
             --download=False \
             --eval_static=True \
             --mul_rank_size=${RANK_SIZE} \
             --mul_device_id=${RANK_ID} \
             --data_dir=$data_path >$cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait

##############结束训练##########
end=$(date +%s)
e2etime=$(( $end - $start ))


#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
TrainingTime=`grep ,time: $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $4}' | awk -F ':' '{print $2}' | tail -n 1`
wait
FPS=`grep imgs/s $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print '${RANK_SIZE}'*$2}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep sparse_categorical_accuracy $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk 'END {print $NF}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"


#精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_static_eval_'perf' 

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=${TrainingTime}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep ,loss: $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $3}' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt 
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
