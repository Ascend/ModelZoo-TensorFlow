#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=8
export RANK_TABLE_FILE=${cur_path}/../configs/rank_table_8p.json
export JOB_ID=10087
RANK_ID_START=0

export NPU_ENABLE_PERF=true
# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数 需要模型审视修改
#网络名称，同目录名称
Network="Keras-MnasNet_ID3518_for_TensorFlow2.X"
#训练batch_size
batch_size=16384
#训练step
#train_steps=1000
#训练epoch
train_epochs=15
log_steps=24
#学习率
#learning_rate=0.000144



#TF2.X独有，需要模型审视修改
#export NPU_LOOP_SIZE=100
#export GE_USE_STATIC_MEMORY=1


#维持参数，不需要修改
precision_mode="allow_mix_precision"
#precision_mode="allow_fp32_to_fp16"
#维持参数，以下不需要修改
over_dump=False
if [[ $over_dump == True ]];then
    over_dump_path=${cur_path}/overflow_dump
    mkdir -p ${over_dump_path}
fi
auto_tune=False
data_dump_flag=False
data_dump_step="10"
profiling=False
use_mixlist=False
mixlist_file=${cur_path}/../configs/ops_info.json
fusion_off_flag=False
fusion_off_file=${cur_path}/../configs/fusion_switch.cfg

if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_8p_32bs.sh <args>"

    echo " "
    echo "parameter explain:
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		   data dump flag, default is 0
    --data_dump_step		   data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --data_path		           source data of training
    -h/--help		           show help message
    "
    exit 1
fi

#参数校验，需要模型审视修改
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
    elif [[ $para == --auto_tune* ]];then
        auto_tune=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

cd $cur_path/..
start_time=$(date +%s)
#############执行训练#########################
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID

    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID
    fi
    
    
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
	nohup python3 train.py \
   --data_dir=${data_path} \
   --epochs=${train_epochs} \
   --batch_size=${batch_size} \
   --log_steps=${log_steps} \
   --static \
   --rank_size $RANK_SIZE \
   --device_id $ASCEND_DEVICE_ID \
   --precision_mode=${precision_mode} \
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
   --auto_tune=${auto_tune} >$cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &

done
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "examples/second" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F " " '{print$4}' | awk '{if(NR!=1){sum+=$NF}}END{printf "%.4f\n", sum/(NR-1)}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=$(grep "accuracy" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F " " 'END{print$9}')
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'static'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=$(awk -v x=1 -v y="$FPS" 'BEGIN{printf "%.8f\n",x/y}')

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'accuracy' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F " " '{print$6}' >$cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=$(awk 'END {print $1}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt)


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
