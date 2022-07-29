#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`/..

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=8
export RANK_TABLE_FILE=${cur_path}/configs/rank_table_8p.json
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数 需要模型审视修改
#网络名称，同目录名称
Network="BERT_ID3225_for_TensorFlow2.X"
#训练batch_size
batch_size=1024
eval_batch_size=16
#训练step
train_steps=11259
steps_per_loop=100
export NPU_LOOP_SIZE=${steps_per_loop}
#训练epoch
train_epochs=5
#学习率
learning_rate=0.00075

#TF2.X独有，需要模型审视修改
#export NPU_LOOP_SIZE=1000
#export GE_USE_STATIC_MEMORY=1

############维测参数##############
precision_mode="allow_fp32_to_fp16"
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
auto_tune=False
fusion_off_file=${cur_path}/configs/fusion_switch.cfg
############维测参数##############

if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_phase1_8p.sh <args>"

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


############维测参数##############
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
    elif [[ $para == --server_index* ]];then
        server_index=`echo ${para#*=}`
    elif [[ $para == --conf_path* ]];then
        conf_path=`echo ${para#*=}`
    elif [[ $para == --devices_num* ]];then
	    devices_num=`echo ${para#*=}`
    fi
done
one_node_ip=`find $conf_path -name "server_111._0.info"|awk -F "server_" '{print $2}'|awk -F "_" '{print $1}'`
linux_num=`find $conf_path -name "server_*.info" |wc -l`
############维测参数##############

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
export batch_size=`awk 'BEGIN{printf "%.0f\n",'1024'*'${linux_num}'}'`
export eval_batch_size=`awk 'BEGIN{printf "%.0f\n",'16'*'${linux_num}'}'`
export RANK_SIZE=`awk 'BEGIN{printf "%.0f\n",'${devices_num}'*'${linux_num}'}'`
rank_size=8
nohup python3 set_ranktable.py --npu_nums=$linux_num --conf_path=$conf_path
wait
export RANK_TABLE_FILE=$cur_path/test/rank_table.json
export JOB_ID=10087
export DEVICE_INDEX=0
cd $cur_path

seq_len=128
max_pred_per_seq=20

BERT_CONFIG=${data_path}/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_config.json
INPUT_FILES="${data_path}/tfrecord/seq_len_${seq_len}_max_pred_${max_pred_per_seq}/wikicorpus_en/training/*"
EVAL_FILES="${data_path}/tfrecord/seq_len_${seq_len}_max_pred_${max_pred_per_seq}/wikicorpus_en/test"
#INPUT_FILES="${data_path}/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/training/*"
#EVAL_FILES="${data_path}/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/test"



start_time=$(date +%s)
#############执行训练#########################
for((RANK_ID=$((rank_size*server_index));RANK_ID<$((((server_index+1))*rank_size));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=`expr ${RANK_ID} - $((rank_size*server_index))`
    ASCEND_DEVICE_ID=`expr ${RANK_ID} - $((rank_size*server_index))`
    echo "Device ID: $ASCEND_DEVICE_ID"

    if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt_${learning_rate}
    else
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt_${learning_rate}
    fi
    
    #绑核，不需要绑核的模型删除，需要绑核的模型根据实际修改
    cpucount=`lscpu | grep "CPU(s):" | head -n 1 | awk '{print $2}'`
    cpustep=`expr $cpucount / 8`
    echo "taskset c steps:" $cpustep
    let a=ASCEND_DEVICE_ID*$cpustep
    let b=ASCEND_DEVICE_ID+1
    let c=b*$cpustep-1
    
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    if [ "x${bind_core}" != x ];then
        bind_core="taskset -c $a-$c"
    fi
    nohup ${bind_core} python3 run_pretraining.py \
        --input_files=$INPUT_FILES \
        --eval_files=$EVAL_FILES \
        --model_dir=${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt_${learning_rate} \
        --bert_config_file=$BERT_CONFIG \
        --train_batch_size=$batch_size \
        --eval_batch_size=$eval_batch_size \
        --max_seq_length=$seq_len \
        --max_predictions_per_seq=$max_pred_per_seq \
        --num_steps_per_epoch=${train_steps} \
        --num_train_epochs=${train_epochs} \
        --steps_per_loop=${steps_per_loop} \
        --save_checkpoint_steps=${train_steps} \
        --warmup_steps=15012 \
        --distribution_strategy=one_device \
        --num_gpus=1 \
        --stop_steps=300 \
        --num_accumulation_steps=1 \
        --learning_rate=${learning_rate} \
        --optimizer_type=LAMB \
        --precision_mode=${precision_mode} \
        --use_reducemean=True \
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
        --auto_tune=${auto_tune} \
        --use_fp16 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait
end=$(date +%s)
e2e_time=$(( $end - $start_time ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep  "Total Training Time" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk '{print$(NF-4)}'`
FPS=`grep  "Throughput Average (sentences/sec)" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep -v "with overhead" | awk '{print$(NF-0)}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep  "Validation masked_lm_accuracy" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print$(NF-0)}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"

#精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_BertBase_phase1_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=${TrainingTime}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep  "Train Step:" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F ' ' '{print$11}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt 
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
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log


for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    sed -i "/AttributeError/d" $cur_path/output/${RANK_ID}/train_${RANK_ID}.log
    sed -i "/ModuleNotFoundError/d" $cur_path/output/${RANK_ID}/train_${RANK_ID}.log 
done
