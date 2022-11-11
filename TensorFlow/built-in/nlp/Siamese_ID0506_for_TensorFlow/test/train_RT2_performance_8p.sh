#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`/../

#集合通信参数,不需要修改

export RANK_SIZE=8
export JOB_ID=10087
export RANK_ID=8p
RANK_ID_START=0
export RANK_TABLE_FILE=${cur_path}/test/8p.json
export HCCL_CONNECT_TIMEOUT=600
RANK_SIZE=8

#使能RT2.0
export ENABLE_RUNTIME_V2=1

#export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Siamese_ID0506_for_TensorFlow"
#训练epoch
train_epochs=10 #init1
#训练batch_size
batch_size=64
# 训练step
train_steps=
# 学习率
learning_rate=4e-4


#维测参数，precision_mode需要模型审视修改
#precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False
# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		     data dump flag, default is False
    --data_dump_step		     data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --data_path		           source data of training
    -h/--help		             show help message
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
cd $cur_path
#sed -i "s|/scratch/shiyichu/dataset/FaceDatabases/CASIA-Webface/casia_mtcnncaffe_aligned|${data_path}|g" ./data/list_casia_mtcnncaffe_aligned_nooverlap.txt
#for i in 0 1 2 3 4 5 6 7
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export DEVICE_INDEX=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID
    export DEVICE_ID=$ASCEND_DEVICE_ID
    echo "Device ID: $ASCEND_DEVICE_ID"


    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
    fi

    # corenum=`cat /proc/cpuinfo |grep 'processor' | wc -l`
    # let a=RANK_ID*${corenum}/8
    # let b=RANK_ID+1
    # let c=b*${corenum}/8-1
    # if [ "x${bind_core}" != x]；then
    #     bind_core="taskset -c $a-$c"
    # fi

    # sed -i "s|ind_start = 0 * part_int|ind_start = ${i} * part_int|g" $cur_path/../train.py
    
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
    # timeout -s SIGINT 3600 nohup python3 $cur_path/../train_8p.py \
    nohup python3 ${cur_path}/train_rt.py \
    --num_epochs $train_epochs \
    --training_files=$data_path/person_match.train2 \
    --hidden_units=64 \
    --embedding_dim=304 \
    --device_size=8 \
    --device_id=$RANK_ID \
    --evaluate_every=10000 \
    --checkpoint_every=10000 > ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

    # sed -i "s|ind_start = ${i} * part_int|ind_start = 0 * part_int|g" $cur_path/../train.py


    # sleep 60
    # num=`grep 'ERROR' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep "out of bounds"| wc -l`
    # while [ $num -eq 0 ]
    # do
    #    num=`grep 'ERROR' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep "out of bounds"| wc -l`
    #    echo "${num}"
    #    sleep 5
    # done
    # ps -ef | grep python3 |grep max_epoch |grep config |awk '{system("kill -9 "$2)}'
    # echo "killed Yolov4"

done
wait
#sed -i "s|${data_path}|/scratch/shiyichu/dataset/FaceDatabases/CASIA-Webface/casia_mtcnncaffe_aligned|g" ./data/list_casia_mtcnncaffe_aligned_nooverlap.txt

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
echo "E2E Training Duration sec : $e2e_time"

# 结果打印，不需要修改
TrainingTime=`grep "TRAIN " ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $6}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${TrainingTime}'*1000*'${RANK_SIZE}'}'`
# 打印，不需要修改
echo "Final Performance images/sec: $FPS"


#输出训练精度，需要模型审视修改
train_accuracy=`grep "TRAIN " ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $10}'`

# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'RT2'_'perf'

#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${BatchSize}'/'${FPS}' }'`

grep "TRAIN " ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $8}' > ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt

# 最后一个loss值
ActualLoss=`awk 'END {print $1}' ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt`



#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log














# ###下面字段用于冒烟看护
# BatchSize=${batch_size}
# #设备类型，自动获取
# DeviceType=`uname -m`
# # #用例名称，自动获取
# # CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

# ##获取错误信息
# #系统错误信息
# error_msg="of dimension 1  out of bounds"
# #判断错误信息是否和历史状态一致，此处无需修改
# Status=`grep "${error_msg}" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | wc -l`
# #失败阶段，枚举值图准备FAIL/图拆分FAIL/图优化FAIL/图编译FAIL/图执行FAIL/流程OK
# ModelStatus="图执行FAIL"
# #DTS单号或者issue链接
# DTS_Number="DTS20211112715497"

# #关键信息打印到CaseName.log中，此处无需修改
# echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "BatchSize = ${batch_size}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "ModelStatus = ${ModelStatus}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "DTS_Number = ${DTS_Number}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "Status = ${Status}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "error_msg = ${error_msg}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
