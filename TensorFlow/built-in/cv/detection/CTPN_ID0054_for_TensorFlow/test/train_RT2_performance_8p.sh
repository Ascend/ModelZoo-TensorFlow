#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=8
export RANK_TABLE_FILE=$cur_path/${RANK_SIZE}p.json
export JOB_ID=10087
export DEVICE_INDEX=0
RANK_ID_START=0

#使能RT2.0
export ENABLE_RUNTIME_V2=1

# 数据集路径,保持为空,不需要修改
data_path=""
#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL_ETP=3



Network="CTPN_ID0054_for_TensorFlow"
#训练epoch
#train_epochs=1
#训练batch_size
batch_size=1
#训练step
train_steps=100
save_checkpoint_steps=20
#学习率
learning_rate=8e-5


#TF2.X独有，不需要修改
#export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False


if [[ $1 == --help || $1 == -h ]];then 
    echo "usage: ./train_performance_8p.sh <args>"

    echo ""
    echo "parameter explain:
    --task_name           finetune dataset
    --data_path           source data of training
    --model_path          the path of pretrain ckpt
    --train_batch_size    training batch
    --learning_rate       learning_rate
    --num_train_epochs    epochs
    --output_dir          output dir
    -h/--help             Show help message
    "
    exit 1
fi

if [   -d $cur_path/output ];then
   rm -rf $cur_path/output/*
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
fi


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
    fi
done


if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi


#CTPN独有
cd $cur_path/../
cd utils/bbox/
chmod +x make.sh
./make.sh



#############执行训练#########################
start=$(date +%s)

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi
  
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    cd $cur_path/../
    nohup python3 main/train_npu.py \
        --precision_mode=$precision_mode \
        --pretrained_model_path=$data_path/vgg_16.ckpt \
        --dataset_dir=$data_path \
        --max_steps=$train_steps \
        --device_id=$ASCEND_DEVICE_ID \
        --npu_nums=$RANK_SIZE \
		--DEVICE_ID=$ASCEND_DEVICE_ID \
        --save_checkpoint_steps=$save_checkpoint_steps \
        --checkpoint_path=${cur_path}/output/$ASCEND_DEVICE_ID/ckpt > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait

end=$(date +%s)
e2etime=$(( $end - $start ))




#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
train_time=`grep Step  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $13}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${RANK_SIZE}'*'1000'*'${batch_size}'/'${train_time}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
#输出训练精度,需要模型审视修改
train_accuracy=`grep Calculated $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $6}' | awk -F '}' '{print $1}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"
#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'RT2'_'perf'
##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=$train_time
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "total loss" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $12}' | awk -F ',' '{print $1}'>> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
sed -i "s/ModuleNotFoundError: No module named 'impl.unsorted_segment_sum'/ /g" `grep ModuleNotFoundError -rl $cur_path/output/$ASCEND_DEVICE_ID/train_*.log`




















