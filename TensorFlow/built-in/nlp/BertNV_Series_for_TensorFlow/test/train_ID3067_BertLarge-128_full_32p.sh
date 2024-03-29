#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=8
export JOB_ID=99990001
#export RANK_TABLE_FILE=${cur_path}/../configs/8p.json
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="BertLarge-128_ID3067_for_TensorFlow"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=128
#训练step  1140000 / (128*8/256)
# warmup step 10000 / (128*8/256)
train_steps=71500
#学习率
learning_rate=

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
    echo"usage:./train_full_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		         if or not over detection, default is False
    --data_dump_flag	     data dump flag, default is False
    --data_dump_step		 data dump step, default is 10
    --profiling		         if or not profiling for performance debug, default is False
    --autotune               whether to enable autotune, default is False
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
    elif [[ $para == --server_index* ]];then
        server_index=`echo ${para#*=}`
    elif [[ $para == --conf_path* ]];then
        conf_path=`echo ${para#*=}`
    elif [[ $para == --devices_num* ]];then
        devices_num=`echo ${para#*=}`
    elif [[ $para == --servers_num* ]];then
        servers_num=`echo ${para#*=}`
    fi
done


linux_num=$servers_num


#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

export RANK_SIZE=`awk 'BEGIN{printf "%.0f\n",'${devices_num}'*'${linux_num}'}'`
rank_size=8
nohup python3 set_ranktable.py --npu_nums=$linux_num --conf_path=$conf_path
wait
export RANK_TABLE_FILE=$cur_path/rank_table.json


#训练开始时间，不需要修改
start_time=$(date +%s)
#进入训练脚本目录，需要模型审视修改
#for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
for((RANK_ID=$((rank_size*server_index));RANK_ID<$((((server_index+1))*rank_size));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    #export ASCEND_DEVICE_ID=$RANK_ID
    export ASCEND_DEVICE_ID=`expr ${RANK_ID} - $((rank_size*server_index))`
    ASCEND_DEVICE_ID=`expr ${RANK_ID} - $((rank_size*server_index))`

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt${ASCEND_DEVICE_ID}
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt${ASCEND_DEVICE_ID}
    fi
    
     # 绑核，不需要的绑核的模型删除，需要模型审视修改
    corenum=`cat /proc/cpuinfo |grep "processor"|wc -l`
    let a=RANK_ID*${corenum}/${RANK_SIZE}
    let b=RANK_ID+1
    let c=b*${corenum}/${RANK_SIZE}-1

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
    if [ "x${bind_core}" != x ];then
        bind_core="taskset -c $a-$c"
    fi
    nohup python3.7 ${cur_path}/../src/run_pretraining.py --bert_config_file=${cur_path}/../configs/bert_large_config.json \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --train_batch_size=${batch_size} \
    --learning_rate=1.2e-4 \
    --num_warmup_steps=1200 \
    --num_train_steps=${train_steps} \
    --optimizer_type=adam \
    --manual_fp16=True \
    --use_fp16_cls=True \
    --input_files_dir=${data_path}/tfrecord/seq_len_128_max_pred_20/wikicorpus_en/training \
    --eval_files_dir=${data_path}/tfrecord/seq_len_128_max_pred_20/wikicorpus_en/test \
    --npu_bert_debug=False \
    --npu_bert_use_tdt=True \
    --do_train=True \
    --do_eval=True \
    --num_accumulation_steps=1 \
    --npu_bert_job_start_file= \
    --iterations_per_loop=1000 \
    --save_checkpoints_steps=10000 \
    --npu_bert_clip_by_global_norm=False \
    --distributed=True \
    --npu_bert_tail_optimize=True \
    --npu_bert_loss_scale=0 \
    --over_dump=${over_dump} \
    --over_dump_path=${over_dump_path} \
    --output_dir=${cur_path}/output/${ASCEND_DEVICE_ID}/ckpt${ASCEND_DEVICE_ID} > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
ActualFPS=`grep Throughput ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $6}'`
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}' * '${RANK_SIZE}' / '${ActualFPS}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#输出训练精度,需要模型审视修改
TrainAccuracy=`grep "tensorflow:  masked_lm_accuracy" $cur_path/output/0/train_0.log|awk 'END {print $4}'`
#打印，不需要修改
echo "Final Train Accuracy : ${TrainAccuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'


#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "tensorflow:loss =" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "loss = " '{print $2}' | awk -F "," '{print $1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
echo "TrainAccuracy = ${TrainAccuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
