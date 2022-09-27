#!bin/bash
cur_path=`pwd`

#环境设置，需要根据网络修改
export PYTHONPATH=$cur_path/../models/research:$cur_path/../models/research/slim:$PYTHONPATH
export HCCL_CONNECT_TIMEOUT=200
#集合通信
export RANK_SIZE=8
export RANK_TABLE_FILE=$cur_path/../configs/${RANK_SIZE}p.json
export JOB_ID=10087
RANK_ID_START=0
ASCEND_DEVICE_ID_START=0

#数据集参数
data_path="/data"
use_conda=0

#训练参数，需要根据模型修改
Network="SSD-MobilenetV2_ID0499_for_TensorFlow"
num_train_steps=1000
batch_size=24
ckpt_path=/checkpoints
pipeline_config=$cur_path/../models/research/configs/ssd_mobilenet_v2_coco_8p.config

#帮助提示，需要根据网络修改
if [[ $1 == --help || $1 == -h ]];then 
    echo "usage: ./train_performance_8p.sh <args>"
    echo ""
    echo "parameter explain:
    --num_train_steps           training steps
    --data_path                 source data of training
    --ckpt_path                  pre-checkpoint path
    --pipeline_config           pipeline config path
	--skip_eval  whether to skip eval
    -h/--help                   Show help message
    "
    exit 1
fi

#入参设置，需要根据网络修改
for para in $*
do
    if [[ $para == --num_train_steps* ]];then
		num_train_steps=`echo ${para#*=}`
	elif [[ $para == --data_path* ]];then
		data_path=`echo ${para#*=}`
	elif [[ $para == --ckpt_path* ]];then
		ckpt_path=`echo ${para#*=}`
	elif [[ $para == --pipeline_config* ]];then
		pipeline_config=`echo ${para#*=}`
    elif [[ $para == --use_conda* ]];then
	    use_conda=`echo ${para#*=}`
	elif [[ $para == --skip_eval* ]];then
	    skip_eval=`echo ${para#*=}`
    fi
done	

if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be config"
    exit 1
fi

##########################执行训练#########################
start_time=$(date +%s)
cd $cur_path/../models/research
if [  -f ${pipeline_config}.bak ];then
   cp ${pipeline_config}.bak ${pipeline_config}
else
   cp ${pipeline_config} ${pipeline_config}.bak
fi

# 更改参数
sed -i "s%/data/coco2017_tfrecords%${data_path}/coco2017_tfrecords%p" ${pipeline_config}

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
    do
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$((ASCEND_DEVICE_ID_START+RANK_ID))
    echo "Device ID: $ASCEND_DEVICE_ID"
    if [   -d $cur_path/output/${ASCEND_DEVICE_ID} ];then
        rm -rf $cur_path/output/${ASCEND_DEVICE_ID}
        mkdir -p $cur_path/output/${ASCEND_DEVICE_ID}
    else
        mkdir -p $cur_path/output/${ASCEND_DEVICE_ID}
    fi

    #训练执行脚本，需要根据网络修改
    nohup python3 -u ./object_detection/model_main.py \
        --pipeline_config_path=${pipeline_config} \
        --model_dir=$cur_path/output/${ASCEND_DEVICE_ID}/npu_ckpt_mobilenetv2_${RANK_SIZE}p\
        --data_path=${data_path}   \
        --alsologtostder \
        --amp \
        --skip_eval=True \
        --num_train_steps=${num_train_steps}  \
        "${@:1}"  > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait

end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
echo "Final Training Duration sec : $e2e_time"
ASCEND_DEVICE_ID=0

# 参数回改
sed -i "s%${data_path}/coco2017_tfrecords%/data/coco2017_tfrecords%p" ${pipeline_config}


################################性能结果处理#########################
echo "-----------------------Final result------------------------"
# 性能FPS计算，需要根据网络修改
step_sec=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $2}'|tail -2|head -n 1`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${step_sec}'}'`
echo "Final Performance images/sec : ${FPS}"

#################################精度结果处理#########################
# 精度计算，需要根据网络修改
train_accuracy=`grep Precision $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'NR==1 {print $13}'`
echo "Final Training Accuracy mAP: ${train_accuracy}"

#################################性能看护#############################
# 训练用例信息，不需要修改
DeviceType=`uname -m`
BatchSize=${batch_size}
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
ActualFPS=${FPS}
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${FPS}'}'`

#################################Loss#########################
ASCEND_DEVICE_ID=7
# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型修改
grep INFO:tensorflow:loss $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $3}'|sed 's/,//g'|sed '/^$/d' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
echo "Final Training Accuracy loss: ${ActualLoss}"

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
