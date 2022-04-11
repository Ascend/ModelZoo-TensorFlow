#!/bin/bash
cur_path=`pwd`/../

#基础参数，需要模型审视修改
#Batch Size
batch_size=128
#网络名称，同目录名称
Network="GAN_ID2351_for_TensorFlow2.X"
#Device数量，单卡默认为1
RankSize=1
export RANK_SIZE=1
#训练epoch，可选
train_epochs=400

#参数配置
data_path="./dataset"

# #维测参数，precision_mode需要模型审视修改
# precision_mode="allow_fp32_to_fp16"
# #维持参数，以下不需要修改
# over_dump=False
# data_dump_flag=False
# data_dump_step="10"
# profiling=False

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
use_mixlist=True
mixlist_file="./configs/ops_info.json"
fusion_off_flag=False
fusion_off_file="./configs/fusion_switch.cfg"
############维测参数##############

if [[ $1 == --help || $1 == --h ]];then
    echo"usage:./train_performance_1p.sh <args>"
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

# #参数校验，不需要修改
# for para in $*
# do
#     if [[ $para == --precision_mode* ]];then
#         precision_mode=`echo ${para#*=}`
#     elif [[ $para == --over_dump* ]];then
#         over_dump=`echo ${para#*=}`
#         over_dump_path=${cur_path}/output/overflow_dump
#         mkdir -p ${over_dump_path}
#     elif [[ $para == --data_dump_flag* ]];then
#         data_dump_flag=`echo ${para#*=}`
#         data_dump_path=${cur_path}/output/data_dump
#         mkdir -p ${data_dump_path}
#     elif [[ $para == --data_dump_step* ]];then
#         data_dump_step=`echo ${para#*=}`
#     elif [[ $para == --profiling* ]];then
#         profiling=`echo ${para#*=}`
#         profiling_dump_path=${cur_path}/output/profiling
#         mkdir -p ${profiling_dump_path}
#     elif [[ $para == --data_path* ]];then
#         data_path=`echo ${para#*=}`
#     fi
# done

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
    fi
done
############维测参数##############

##############执行训练##########
cd $cur_path
if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
nohup python3 tf_v2_03_WGAN.py  --data_path=${data_path} \
  --train_epochs=${train_epochs} \
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
	--profiling_dump_path=${profiling_dump_path} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

echo "Final Training Duration sec : $e2etime"
#结果打印，不需要修改
echo "------------------ Final result ------------------"
DeviceType=`uname -m`
CaseName=${Network}_bs${batch_size}_${RankSize}'p'_'acc'

#输出性能FPS，需要模型审视修改
#TrainingTime=`grep "Time=" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $4}'|cut -d = -f 2`
TrainingTime=`grep "Time" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk 'END{print $6}'`
wait
ActualFPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${TrainingTime}'}'`
train_accuracy=`grep -a 'd_loss:' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log | awk 'END{print $2}'`

##获取性能数据，不需要修改
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "d_loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $2}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
