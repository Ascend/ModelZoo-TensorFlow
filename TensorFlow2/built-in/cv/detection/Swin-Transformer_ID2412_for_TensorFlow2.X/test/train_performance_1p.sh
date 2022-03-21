#!/bin/bash
cur_path=`pwd`
#失败用例打屏

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Swin-Transformer_ID2412_for_TensorFlow2.X"
#Device数量，单卡默认为1
RankSize=1
#Batch Size
batch_size=16
#训练epoch，可选
epochs=3
#训练step
train_steps=10

############维测参数##############
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
if [[ $over_dump == True ]];then
    over_dump_path=$cur_path/overflow_dump #此处cur_path为代码根目录
    mkdir -p ${over_dump_path}
fi
data_dump_flag=False
data_dump_step="10"
profiling=False
use_mixlist=True
mixlist_file="../configs/ops_info.json"
fusion_off_flag=False
fusion_off_file="../configs/fusion_switch.cfg"
############维测参数##############

#参数配置
data_path=""
ckpt_path=""
model_name=""

if [[ $1 == --help || $1 == --h ]];then
    echo"usage:./train_full_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                 if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step         data dump step, default is 10
    --profiling                 if or not profiling for performance debug, default is False
    --autotune               whether to enable autotune, default is False
    --data_path                 source data of training
    --ckpt_path                 source ckpt of training
    -h/--help                 show help message
    "
   exit 1
fi

##############执行训练##########
cd $cur_path
if [ -d $cur_path/output ];then
   rm -rf $cur_path/output/*
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
fi
wait

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
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
elif [[ $ckpt_path == "" ]];then
   echo "[Error] para \"ckpt_path\" must be config"
   exit 1
fi


start=$(date +%s)
nohup python3 ../swintransformer/train_main.py \
        --data_path=${data_path} \
        --ckpt_path=${ckpt_path} \
        --model_name="swin_large_224" \
        --epochs=${epochs} \
        --train_steps=${train_steps} \
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
        --profiling_dump_path=${profiling_dump_path}} > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"

###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
cp -f $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log.bak
sed -i 's/\x0d/\n/g' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log.bak
single_batch_step_sec=`grep ms/step  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log.bak | awk 'END {print $5}' | awk -F 'm' '{print $1}'`
FPS=`echo ${single_batch_step_sec} ${batch_size} | awk '{print $2 * 1000 / $1}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

grep "ms/step" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log.bak|awk '{print$8}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
#输出训练精度,需要模型审视修改
grep "f1 score" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_acc.txt
train_accuracy=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_acc.txt |awk '{print $3}' |awk -F "," '{print $1}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*'${RankSize}'*1000/'${FPS}'}'`

##获取错误信息
#系统错误信息
error_msg="the shape of grad must equal with var"
#判断错误信息是否和历史状态一致，此处无需修改
Status=`grep "${error_msg}" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | wc -l`
#失败阶段，枚举值图准备FAIL/图拆分FAIL/图优化FAIL/图编译FAIL/图执行FAIL/流程OK
ModelStatus="流程OK"
#DTS单号或者issue链接
DTS_Number=""

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
