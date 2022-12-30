#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

#基础参数，需要模型审视修改
#Batch Size
batch_size=32
#网络名称，同目录名称
Network="Swin-Transformer_ID2374_for_TensorFlow2.X"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=
#训练step
train_steps=
#学习率
learning_rate=0.01

#参数配置
data_path=""

############维测参数##############
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
if [[ $over_dump == True ]];then
    over_dump_path=$cur_path/test/overflow_dump #此处cur_path为代码根目录
    mkdir -p ${over_dump_path}
fi
data_dump_flag=False
data_dump_step="10"
profiling=False
use_mixlist=False
mixlist_file="./configs/ops_info.json"
fusion_off_flag=False
fusion_off_file="./configs/fusion_switch.cfg"
############维测参数##############

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_full_1p.sh "
   exit 1
fi

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
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
    fi
done

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path\" must be config"
   exit 1
fi
##############执行训练##########
cd $cur_path
if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

#拷贝并修改数据集txt文件
cp -r ${data_path}/cifar-100-python /root/.keras/datasets/

start=$(date +%s)
nohup python3 swin_transformers.py \
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
        --profiling_dump_path=${profiling_dump_path}} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep "352/352" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F '352/352' '{print $2}'|grep -v 'ETA'|grep 'loss:'|awk 'END {print $4}'|cut -d 'm' -f -1`


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'RT2'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=`awk 'BEGIN{printf "%.2f\n", '1000'*'${batch_size}'/'${TrainingTime}'}'`

#获取模型精度
train_accuracy=`grep "352/352" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F '352/352' '{print $2}'|grep 'loss:'|awk 'END {print $10}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "352/352" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F '352/352' '{print $2}'|grep -v 'ETA'|grep 'loss:'|awk '{print $7}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log