#!/bin/bash


# export ASCEND_SLOG_PRINT_TO_STDOUT=1 
# export ASCEND_GLOBAL_LOG_LEVEL=0 # debug level
# export DUMP_GE_GRAPH=3
# export DUMP_GRAPH_PATH='/home/jiayan/avod/avod_npu_20210604062633/ge_graph_tmp'

export EXPERIMENTAL_DYNAMIC_PARTITION=1

cur_path=`pwd`/../

#集合通信参数，不需要修改
export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0
#ASCEND_DEVICE_ID=$VISIBLE_IDS
#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改

#网络名称，同目录名称
Network="AVOD_ID0818_for_TensorFlow"

#batch_size

#训练epoch，可选
# train_epochs=1
#训练step
train_steps=100
checkpoint_interval=100
summary_interval=1 # same as batch size

#学习率; 每 30000 steps，衰减到0.8 -> 8*-5
initial_learning_rate=1e-4

#参数配置
data_path=""

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
profiling=0 # False or 1 for True
profiling_dump_path=${cur_path}/test/output/profiling

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
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
    elif [[ $para == --checkpoint_interval* ]];then
        checkpoint_interval=`echo ${para#*=}`
    elif [[ $para == --summary_interval* ]];then
        summary_interval=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/test/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# if [[ $data_path  == "" ]];then
#    echo "[Error] para \"data_path \" must be config"
#    exit 1
# fi

##############执行训练##########
# cd $cur_path/../
#sed -i "s|dataset/|${data_path}/dataset/|g" config.py
#sed -i "s|EPOCHS = 1|EPOCHS = ${train_epochs}|g" config.py
#wait


start=$(date +%s)
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export DEVICE_INDEX=${ASCEND_DEVICE_ID}

    export RANK_ID=$RANK_ID
    
    #创建DeviceID输出目录，不需要修改: Found at avod/data/outputs/CONFIG_NAME/checkpoints
    if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
        # rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt_model
    else
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt_model
    fi
    

    pwd 
    echo "Training..."
    echo "Results log is saved at ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log"
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    cd $cur_path
    export PYTHONPATH=$PYTHONPATH:${cur_path}
    export PYTHONPATH=$PYTHONPATH:${cur_path}'/wavedata'
    echo $cur_path
    sed -i "s|/home/jiayan/avod/Kitti/object|${data_path}/Kitti/object|g" avod/protos/kitti_dataset.proto
    sh scripts/install/build_integral_image_lib.bash
    sh avod/protos/run_protoc.sh
    python3 scripts/preprocessing/gen_mini_batches.py
    cd $cur_path
    nohup python3 avod/experiments/run_training.py \
      --pipeline_config=avod/configs/pyramid_cars_with_aug_example.config \
      --train_steps=${train_steps} \
      --checkpoint_interval=${checkpoint_interval} \
      --summary_interval=${summary_interval} \
      --initial_learning_rate=${initial_learning_rate} \
      --precision_mode=${precision_mode} \
      --profiling=${profiling} \
      --profiling_dump_path=${profiling_dump_path} > ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    sed -i "s|${data_path}/Kitti/object|/home/jiayan/avod/Kitti/object|g" avod/protos/kitti_dataset.proto
done 
wait

#训练结束时间，不需要修改
end=$(date +%s)
e2e_time=$(( $end - $start ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

#参数回改
#sed -i "s|${data_path}/dataset/|dataset/|g" config.py
#sed -i "s|EPOCHS = ${train_epochs}|EPOCHS = 1|g" config.py

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep "Time Elapsed" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $8}'|tail -n +3|awk '{sum+=$1} END {print  sum/NR}'`
wait
FPS=`awk 'BEGIN{printf "%.3f\n",'${summary_interval}'/'${TrainingTime}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
# train_accuracy=`grep "valid accuracy:" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $14}'`
#打印，不需要修改
# echo "Final Train Accuracy : ${train_accuracy}"


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${summary_interval}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Total Loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $5}'|awk -F , '{print $1}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
