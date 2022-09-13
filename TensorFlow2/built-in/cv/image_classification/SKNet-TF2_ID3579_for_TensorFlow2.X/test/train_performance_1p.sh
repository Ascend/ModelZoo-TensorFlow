#!/bin/bash
cur_path=$(pwd)/../

#基础参数，需要模型审视修改
#Batch Size
batch_size=32
#网络名称，同目录名称
Network="SKNet-TF2_ID3579_for_TensorFlow2.X"
#Device数量，单卡默认为1
RANK_SIZE=1
#训练epoch，可选
train_epochs=20


#训练step
#train_steps=50000
#学习率
#learning_rate=1e-5
log_steps=300 # 0.5 * steps_per_epoch
#参数配置
data_path=""

############维测参数##############
precision_mode="allow_mix_precision"
#precision_mode="allow_fp32_to_fp16"
#维持参数，以下不需要修改
over_dump=False
if [[ $over_dump == True ]]; then
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
auto_tune=False
############维测参数##############

if [[ $1 == --help || $1 == --h ]]; then
   echo "usage:./train_performance_1p.sh"
   exit 1
fi

for para in $*; do
   if [[ $para == --data_path* ]]; then
      data_path=$(echo ${para#*=})
   elif [[ $para == --batch_size* ]]; then
      batch_size=$(echo ${para#*=})
   elif [[ $para == --train_epochs* ]]; then
      train_epochs=$(echo ${para#*=})
   elif [[ $para == --log_steps* ]]; then
      log_steps=$(echo ${para#*=})
   elif [[ $para == --precision_mode* ]]; then
      precision_mode=$(echo ${para#*=})
   elif [[ $para == --over_dump* ]]; then
      over_dump=$(echo ${para#*=})
      over_dump_path=${cur_path}/output/overflow_dump
      mkdir -p ${over_dump_path}
   elif [[ $para == --data_dump_flag* ]]; then
      data_dump_flag=$(echo ${para#*=})
      data_dump_path=${cur_path}/output/data_dump
      mkdir -p ${data_dump_path}
   elif [[ $para == --data_dump_step* ]]; then
      data_dump_step=$(echo ${para#*=})
   elif [[ $para == --profiling* ]]; then
      profiling=$(echo ${para#*=})
      profiling_dump_path=${cur_path}/output/profiling
      mkdir -p ${profiling_dump_path}
   elif [[ $para == --use_mixlist* ]]; then
      use_mixlist=$(echo ${para#*=})
   elif [[ $para == --mixlist_file* ]]; then
      mixlist_file=$(echo ${para#*=})
   elif [[ $para == --fusion_off_flag* ]]; then
      fusion_off_flag=$(echo ${para#*=})
   elif [[ $para == --fusion_off_file* ]]; then
      fusion_off_file=$(echo ${para#*=})
   elif [[ $para == --auto_tune* ]]; then
      auto_tune=$(echo ${para#*=})
   fi
done

if [[ $data_path == "" ]]; then
   echo "[Error] para \"data_path \" must be config"
   exit 1
fi

##############执行训练##########
cd $cur_path

if [ -d $cur_path/test/output ]; then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
nohup python3 train.py \
   --data_path=$data_path \
   --epochs=$train_epochs \
   --batch_size=$batch_size \
   --log_steps=${log_steps} \
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
   --auto_tune=${auto_tune} >$cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait

end=$(date +%s)
e2e_time=$(($end - $start))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"
#输出性能FPS，需要模型审视修改
sum_FPS=$(grep 'TimeHistory:' $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | tail -n 5 | awk '{sum += $4};END {print sum}')
num_FPS=$(grep 'TimeHistory:' $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | tail -n 5 | wc -l)
FPS=$(awk 'BEGIN{printf "%2.f\n",'${sum_FPS}'/'${num_FPS}'}')
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=$(grep "1875/1875 - .* - loss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk 'END {print $NF}')
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=$(awk -v x=1 -v y="$FPS" 'BEGIN{printf "%.8f\n",x/y}')

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "1875/1875 - .* - loss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $6}' >$cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=$(awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >$cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>$cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>$cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>$cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>$cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>$cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>$cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >>$cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>$cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>$cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log


