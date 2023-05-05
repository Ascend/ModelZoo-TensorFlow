#!/bin/bash

cur_path=`pwd`/..
#失败用例打屏
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#基础参数，需要模型审视修改
#Batch Size
batch_size=32
#网络名称，同目录名称
Network="bert-squad_ID1566_for_TensorFlow2.X"
#Device数量，单卡默认为1
RankSize=8
RANK_ID_START=0
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=8
export RANK_TABLE_FILE=${cur_path}/configs/rank_table_8p.json
#性能优化
export NPU_LOOP_SIZE=25
#训练epoch，可选
train_epochs=2
#学习率
learning_rate=6.4e-4
ckpt_path=""
#参数配置

# 数据集路径,保持为空,不需要修改
data_path=""
############维测参数##############
precision_mode="allow_mix_precision"
# precision_mode="allow_fp32_to_fp16"
#维持参数，以下不需要修改
over_dump=False
if [[ $over_dump == True ]];then
    over_dump_path=${cur_path}/test/overflow_dump
    mkdir -p ${over_dump_path}
fi
data_dump_flag=False
data_dump_step="10"
profiling=False
use_mixlist=True
mixlist_file=${cur_path}/configs/ops_info.json
fusion_off_flag=False
fusion_off_file=${cur_path}/configs/fusion_switch.cfg
auto_tune=False
############维测参数##############


if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh $data_path --work_dir="$cur_path/estimator_working_dir" --export_path="$cur_path/outputs/models/000001-first_generation""
   exit 1
fi

############维测参数##############
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}test/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}test/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}test/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
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


if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path\" must be config"
   exit 1
fi

##############执行训练##########
cd $cur_path

export PYTHONPATH="$PYTHONPATH:$cur_path"
SQUAD_VERSION=v1.1
BERT_BASE_DIR=${data_path}/pretrained_weights/uncased_L-12_H-768_A-12/
SQUAD_DIR=${data_path}/SQuAD/v1.1
MODE=train_and_eval
#MODE=train
start=$(date +%s)

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID

    if [ -d $cur_path/test/output ];then
        rm -rf $cur_path/test/output/$ASCEND_DEVICE_ID
        mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
    else
        mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
    fi

    nohup python3 ./official/nlp/bert/run_squad.py \
        --mode=${MODE} \
        --input_meta_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_meta_data \
        --train_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
        --predict_file=${SQUAD_DIR}/dev-${SQUAD_VERSION}.json \
        --vocab_file=${BERT_BASE_DIR}/vocab.txt \
        --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
        --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
        --train_batch_size=${batch_size} \
        --optimizer_type=lamb \
        --learning_rate=${learning_rate} \
        --num_train_epochs=${train_epochs} \
        --model_dir=$cur_path/test/output/$ASCEND_DEVICE_ID/ckpt \
        --log_steps=200 \
        --steps_per_loop=${NPU_LOOP_SIZE} \
        --num_gpus=1 \
        --distribution_strategy=one_device \
        --sub_model_export_name=sub_model \
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
done
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep TimeHistory $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk 'END{print $8}'`
wait

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep SQuAD $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $8}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_squad1.1_base_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
trainingTime=`grep TimeHistory $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk 'END{print $6}'`
TrainingTime=`awk 'BEGIN{printf "%.2f",'$trainingTime'/10}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'Train Step' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $11}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk '{print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt|tail -n 1`

echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log