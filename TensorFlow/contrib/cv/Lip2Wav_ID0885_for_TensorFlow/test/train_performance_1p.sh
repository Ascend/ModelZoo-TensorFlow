#!/bin/bash
cur_path=`pwd`/../
echo $cur_path

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0


#export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改

#网络名称，同目录名称
Network="Lip2Wav_ID0885_for_TensorFlow"

#batch_size
batch_size=24 #16

#训练epoch，可选
train_epochs=1
#训练step
train_steps=50
#学习率
learning_rate=1e-5

#参数配置
data_path=""

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

# if [[ $data_path  == "" ]];then
#    echo "[Error] para \"data_path \" must be config"
#    exit 1
# fi

##############执行训练##########

cd $cur_path/main
#sed -i "s|dataset/|${data_path}/dataset/|g" config.py
#sed -i "s|EPOCHS = 1|EPOCHS = ${train_epochs}|g" config.py
#wait

start=$(date +%s)
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt_model
    else
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt_model
    fi
    echo ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log


    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    nohup python3 -u train_forward_test.py "test_run" \
        --data_root $data_path/chess --preset synthesizer/presets/chess.json \
            --tacotron_train_steps 50 > ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &  #/home/jiayansuo/Lip2wav_train/Lip2Wav-master_npu_20210602003231/Dataset/chess/ \ 
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
TrainingTime=`grep "loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|tail -1| awk -F "]" '{print $1}'| awk '{print $8}' | grep -o '[0-9.]\+'` #-F "("
wait

echo "Training Time : $TrainingTime"

FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${TrainingTime}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_loss=`grep "loss=" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| tail -1 | awk '{print $6}' | grep -o '[0-9.]\+'` #awk '{print $5}'` #'{for(i=13;i<18;i++) printf("%s ",$i);}'`
#打印，不需要修改
echo "Final Train Loss : ${train_loss}"


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "avg_loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk '{print ($5)}'| grep -o '[0-9.]\+'> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt # #awk  '{print ($14+$16)}'

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
