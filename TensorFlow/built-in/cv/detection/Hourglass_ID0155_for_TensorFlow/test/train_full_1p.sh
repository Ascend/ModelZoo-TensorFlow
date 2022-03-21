#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Hourglass_ID0155_for_TensorFlow"
#训练batch_size
batch_size=4
#训练step
train_epoch=200
#学习率
learning_rate=0.00025
#其他参数
#max_seq_length=128
#save_steps=600


#TF2.X独有，不需要修改
#export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维测参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
#其他维测参数
#random_remove=False
#loss_scale_flag=False
#loss_scale_value=1

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		         if or not over detection, default is False
    --data_dump_flag	     data dump flag, default is False
    --data_dump_step		 data dump step, default is 10
    --profiling		         if or not profiling for performance debug, default is False
    --data_path		         source data of training
    -h/--help		         show help message
    "
    exit 1
fi

#参数校验，不需要修改
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
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)


#进入训练脚本目录，需要模型审视修改
cd $cur_path/..
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
	echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID


    #创建DeviceID输出目录，不需要修改
    if [ -d $cur_path/test/output ];then
        rm -rf $cur_path/test/output/*
		mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
  	else
		mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

   config_file=$cur_path/../config.cfg
   image_dir=$data_path/images
   dataset_dir=$data_path/dataset.txt
   log_dir=${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
   cp -r $config_file $cur_path/../config_bak_for_training.cfg 
   sed -i "s?./images?$image_dir?g" $config_file
   sed -i "s?./hourglass_tiny?$log_dir?g" $config_file
   sed -i "s?dataset.txt?$dataset_dir?g" $config_file
   nohup python3 train_launcher.py \
   --precision_mode=${precision_mode} > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait  

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#参数回改
sed -i "s?$image_dir?./images?g" $config_file
sed -i "s?$log_dir?./hourglass_tiny?g" $config_file
sed -i "s?$dataset_dir?dataset.txt?g" $config_file

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep -a '100.%' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{FS="-avg_time/batch: "} END {print $2}'|awk '{print $1}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${TrainingTime}'}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "Accuracy" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | cut -d "=" -f 2 | cut -d " " -f 2 | tail -n 1`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`echo "$TrainingTime" | awk '{printf("%.3f\n",$1*1000)}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -a "100.%" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{FS="avg_loss: "} {print $1001}'|awk '{print $1}'|awk 'NR == 1 {next} {print $1}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
