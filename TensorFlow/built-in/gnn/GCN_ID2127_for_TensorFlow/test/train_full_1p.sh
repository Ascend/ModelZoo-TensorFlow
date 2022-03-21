#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`/../

#集合通信参数,不需要修改
export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0


# 数据集路径,保持为空,不需要修改
data_path="./data"

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="GCN_ID2127_for_TensorFlow"

# config_file=res50_32bs_1p_host_full
max_train_steps=200
# iterations_per_loop=1000
# debug=True
# eval=True
#参考config
batch_size=1
#export DEVICE_ID=${RANK_ID_START}
#export ASCEND_DEVICE_ID=${DEVICE_ID}
#DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 1))
#export DEVICE_INDEX=${DEVICE_INDEX}

#TF2.X独有，需要模型审视修改
# export NPU_LOOP_SIZE=${train_steps}

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
        over_dump_path=${cur_path}/test/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/test/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/test/output/profiling
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
cd $cur_path/..
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
    fi
    # 绑核，不需要的绑核的模型删除，需要的模型审视修改
    let a=RANK_ID*12
    let b=RANK_ID+1
    let c=b*12-1
    
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
    cd ${cur_path}
	nohup python3.7 gcn/train.py \
        --data_path ${data_path} \
        --device npu \
        --cora_full \
        --take_subgraphs \
        --min_train_samples 20 \
        --min_valid_samples 30 \
        --valid_size 0 \
        --test_size 99999 \
        --num_epochs ${max_train_steps} \
        --patience 200 \
        --hidden_dim 64 \
        --learning_rate 2e-3 \
        --keep_prob 0.2 \
        --l2_regularizer 2e-5 \
        --shuffle \
        --sparse_adj \
        --sparse_input \
        --pb_file constant_graph_corafull.pb \
        --out_path=${cur_path}/test/output \
        > ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &  
done 
wait

# Create an archive containing model file and input data files
#DATASET=corafull . scripts/prepare_inference_data.sh

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
sec_per_step=`grep "sec" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | tail -1 | cut -d" " -f 15`
FPS=`awk 'BEGIN{printf "%d\n",'${batch_size}'/'${sec_per_step}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
#训练精度，需要从train_$ASCEND_DEVICE_ID.log里，通过关键字获取。需要模型审视修改
train_accuracy=`grep "test_acc" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | tail -1 | cut -d" " -f 11`
echo "Final Train Accuracy: ${train_accuracy}"
#E2E训练端到端时长，直接计算，不需要修改
echo "E2E Training Duration sec: $e2e_time"

# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# #获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'/'${FPS}'}'`
#TrainingTime=`grep "Training ended" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | tail -1 | cut -d" " -f 8`
#ActualTrainSteps=`grep "train_loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | tail -1 | cut -d" " -f 2`
#TrainingTime=`awk "BEGIN {print $TrainingTime/$ActualTrainSteps}"`
#获取Loss，通过train_*.log中关键字，需要根据模型审视
grep "train_loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | cut -d" " -f 7 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`grep "train_loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | tail -1 | cut -d" " -f 7`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
