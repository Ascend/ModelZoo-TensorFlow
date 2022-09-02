#!/bin/bash
cur_path=`pwd`/../

#基础参数，需要模型审视修改
#Batch Size
batch_size=128
#网络名称，同目录名称
Network="sensor-har_ID3399_for_TensorFlow2.X"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=2
#训练step
#train_steps=5
#学习率
learning_rate=0.001
#ckpt_path=""
#参数配置
data_path=""

############维测参数##############
precision_mode="allow_mix_precision"
# precision_mode="allow_fp32_to_fp16"
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
############维测参数##############
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
    elif [[ $para == --auto_tune* ]];then
        auto_tune=`echo ${para#*=}`
    fi
done
############维测参数##############


if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh $data_path"
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   fi
   if [[ $para == --ckpt_path* ]];then
      ckpt_path=`echo ${para#*=}`
   fi
done

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path\" must be config"
   exit 1
fi

##############执行训练##########
cd $cur_path/
if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
nohup python3 main.py   --train \
                        --epochs=$train_epochs \
                        --dataset=pamap2 \
                        --data_path=$data_path \
                        --batch_size=$batch_size \
                        --precision_mode=${precision_mode} \
                        --over_dump=${over_dump} \
                        --static \
                        --over_dump_path=${over_dump_path} \
                        --data_dump_flag=${data_dump_flag} \
                        --data_dump_step=${data_dump_step} \
                        --data_dump_path=${data_dump_path} \
                        --profiling=${profiling} \
                        --use_mixlist=${use_mixlist} \
                        --fusion_off_flag=${fusion_off_flag} \
                        --mixlist_file=${mixlist_file} \
                        --fusion_off_file=${fusion_off_file} \
                        --auto_tune=${auto_tune} \
                        --profiling_dump_path=${profiling_dump_path} >$cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
#echo "Final Training Duration sec : $e2etime"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep TimeHistory: $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log  | awk 'END{print $4}'`
# FPS=`grep TimeHistory: $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log  | awk '{sum+=$4}END{printf "%.4f\n", sum/NR}'`
#输出FPS
# single_Time=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'/'${FPS}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep 231/231 $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk 'END{print $NF}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "Final Training Duration sec : $e2etime"


###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'static'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%f\n",'1'/'${FPS}'}'`
# TrainingTime=${single_Time}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 231/231 $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $(NF-9)}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`


##获取错误信息
#系统错误信息
#ModelStatus="图执行FAIL"
# error_msg="EZ3002"
#判断错误信息是否和历史状态一致，此处无需修改
#error_msg="Graph engine process graph failed: EZ3002: Optype \[Conv2DBackpropFilter\] of Ops kernel"
#Status=`grep "${error_msg}" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|wc -l`
#DTS单号或者issue链接
#DTS_Number="DTS2021090622224"

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "ModelStatus = ${ModelStatus}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "DTS_Number = ${DTS_Number}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "Status = ${Status}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "error_msg = ${error_msg}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
