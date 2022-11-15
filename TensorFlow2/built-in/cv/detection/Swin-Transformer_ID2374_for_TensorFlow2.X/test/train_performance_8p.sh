#!/bin/bash

cur_path=`pwd`/..
#失败用例打屏

#export DUMP_GRAPH_PATH=/home/dump_graph
#export DUMP_GE_GRAPH=2
#export DUMP_GRAPH_LEVEL=3

export ASCEND_GLOBAL_LOG_LEVEL=3
/usr/local/Ascend/driver/tools/msnpureport -g ERROR -d 0
/usr/local/Ascend/driver/tools/msnpureport -g ERROR -d 1
/usr/local/Ascend/driver/tools/msnpureport -g ERROR -d 2
/usr/local/Ascend/driver/tools/msnpureport -g ERROR -d 3
/usr/local/Ascend/driver/tools/msnpureport -g ERROR -d 4
/usr/local/Ascend/driver/tools/msnpureport -g ERROR -d 5
/usr/local/Ascend/driver/tools/msnpureport -g ERROR -d 6
/usr/local/Ascend/driver/tools/msnpureport -g ERROR -d 7

export RANK_SIZE=8
export RANK_TABLE_FILE=${cur_path}/test/rank_table_8p.json
export JOB_ID=10087
RANK_ID_START=0
#export ASCEND_DEVICE_ID=1
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
#基础参数，需要模型审视修改
#Batch Size
batch_size=1024
#网络名称，同目录名称
Network="Swin-Transformer_ID2374_for_TensorFlow2.X"
#Device数量，单卡默认为1
RankSize=8
#训练epoch，可选
train_epochs=5
#训练step
train_steps=
#学习率
learning_rate=0.01

#参数配置
data_path="1"

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
   echo "usage:./train_performance_1p.sh "
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

#拷贝并修改数据集txt文件
cp -r ${data_path}/cifar-100-python /root/.keras/datasets/

start=$(date +%s)
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID
    

    if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt_${learning_rate}
    else
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt_${learning_rate}
    fi
    
#    export DUMP_GRAPH_PATH=test/output/${RANK_ID}/dump_graph_${RANK_ID}
       
    nohup python3 swin_transformers.py --epochs=${train_epochs}  --batch_size=${batch_size} \
        --rank_size=${RANK_SIZE} \
        --device_id=${RANK_ID} \
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
done
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep "44/44" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F '44/44' '{print $2}'|grep -v 'ETA'|grep 'loss:'|awk 'END {print $4}'|cut -d 'm' -f -1`

###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

#吞吐量
ActualFPS=`awk 'BEGIN{printf "%.2f\n", '1000'*'${batch_size}'/'${TrainingTime}'}'`

#获取模型精度
train_accuracy=`grep "44/44" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F '44/44' '{print $2}'|grep 'loss:'|awk 'END {print $10}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "44/44" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F '44/44' '{print $2}'|grep -v 'ETA'|grep 'loss:'|awk '{print $7}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
