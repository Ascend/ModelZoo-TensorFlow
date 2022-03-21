#!/bin/bash


export RANK_SIZE=1
export JOB_ID=10087
export RANK_ID_START=0
export LD_PRELOAD=${LD_PRELOAD}:/usr/local/python3.7.5/lib/python3.7/site-packages/rawpy.libs/libgomp-d22c30c5.so.1.0.0


cur_path=`pwd`
data_path=""
ckpt_path=""
#网络名称，同目录名称
Network="LEARNING-TO-SEE-IN-THE-DARK_ID2069_for_TensorFlow"
#训练epoch
train_epochs=4001
#训练batch_size
batch_size=512
# train_performance_1p.sh perf
# train_full_1p.sh acc
CaseName="${Network}_bs${batch_size}_${RANK_SIZE}p_acc"

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"
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
cd $cur_path/../
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    # mkdir -p train_result_Sony
    # mkdir -p result_Sony/final
    
    nohup python3 -u train_Sony.py \
	  --input_dir=${data_path}/Sony/short/ \
	  --gt_dir=${data_path}/Sony/long/ \
	  --epochs=${train_epochs} > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    wait
    mv result_Sony train_result_Sony

    nohup python3 -u test_Sony.py \
	  --input_dir=${data_path}/Sony/short/ \
	  --gt_dir=${data_path}/Sony/long/ \
	  --checkpoint_dir=train_result_Sony >> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    wait

    nohup python3 -u eval.py \
	  --gt_dir=${data_path}/Sony/long/ \
	  --pic_dir=result_Sony/final/ >> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))



#结果打印，不需要修改
echo "------------------ Final result ------------------"
#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
# getFPS
#输出性能FPS，需要模型审视修改
TrainingTime=`grep "Time= " ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "Time= " '{print $2}'|awk -F " " '{print $1}'|tail -n +3|awk '{sum+=$1} END {print"",sum/NR}'|sed 's/ //g'`
##获取性能数据，不需要修改
#吞吐量
ActualFPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${TrainingTime}'}'`
# getAcc
TrainAccuracy=`grep psnr ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F'psnr:  ' '{print $2}'`
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'Loss= ' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk  '{print $3}'  > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}"              > ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}"          >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${BatchSize}"         >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}"       >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}"           >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}"         >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}"   >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}"       >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${TrainAccuracy}" >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}"    >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
