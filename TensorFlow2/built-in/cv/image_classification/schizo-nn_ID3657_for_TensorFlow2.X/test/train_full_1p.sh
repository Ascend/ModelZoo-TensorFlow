#!/bin/bash
cur_path=`pwd`/../

#设置默认日志级别,不需要修改
# export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#Batch Size
batch_size=32
#网络名称，同目录名称
Network="schizo-nn_ID3657_for_TensorFlow2.X"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=30
#训练step
# train_steps=5
#学习率
ckpt_path=""
#参数配置
data_path=""
#每个epoch打印日志次数
log_steps=125

############维测参数##############
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
if [[ $over_dump == True ]];then
    over_dump_path=${cur_path}/test/overflow_dump
    mkdir -p ${over_dump_path}
fi
data_dump_flag=False
data_dump_step="10"
profiling=False
use_mixlist=False
mixlist_file=${cur_path}/configs/ops_info.json
fusion_off_flag=False
fusion_off_file=${cur_path}/configs/fusion_switch.cfg
auto_tune=False
############维测参数##############

if [[ $1 == --help || $1 == --h ]];then
   echo "usage: ./train_full_1p.sh"
   exit 1
fi

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
    elif [[ $para == --log_steps* ]];then
        log_steps=`echo ${para#*=}`
    elif [[ $para == --auto_tune* ]];then
        auto_tune=`echo ${para#*=}`
    fi
done
############维测参数##############

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path \" must be config"
   exit 1
fi

##############执行训练##########
cd $cur_path

#参数修改
#sed -i "s|../data/tfrecord|${data_path}/data/tfrecord|g" ${cur_path}/data/io/read_tfrecord.py
#sed -i "s|PRETRAINED_CKPT = ROOT_PATH + '/|PRETRAINED_CKPT = '${cur_path}/|g" ${cur_path}/libs/configs/cfgs.py


if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
python3 train.py --data_path=${data_path}\
    --epochs=${train_epochs}\
    --batch_size=${batch_size} \
    --log_steps=$log_steps \
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
                           --auto_tune=${auto_tune} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait

end=$(date +%s)
e2e_time=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
wait
#输出性能FPS，需要模型审视修改
single_batch_step_sec=`grep TimeHistory  $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F " " '{print$4}' | awk 'BEGIN{count=0}{if(NR%15!=1){sum+=$NF;count+=1}}END{printf "%.4f\n", sum/count}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${single_batch_step_sec}'}'`
# TIME=`grep 1936/1936 $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk 'END{print $(NF-6)}' | sed 's/ms\/step//'`
# FPS=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'/'${TIME}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep 'loss: ' $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END{print $NF}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"


#精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%f\n", '1'/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'loss: ' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $6}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
# grep 1936/1936 $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $(NF-3)}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
# train_accuracy=${ActualLoss}

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log