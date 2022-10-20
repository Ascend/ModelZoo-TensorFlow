#!/bin/bash
cur_path=`pwd`/../

RANK_ID_START=0
#基础参数，需要模型审视修改
batch_size=9
#网络名称，同目录名称
Network="SRNTT-l2_ID0272_for_TensorFlow"
#Device数量
export JOB_ID=10001
export RANK_SIZE=8
export RANK_TABLE_FILE=${cur_path}/configs/rank_table_8p.json
#训练epoch，可选
train_epochs=30
#训练step
train_steps=50000
#学习率
learning_rate=1e-4

#参数配置
data_path="./"

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_full_1p.sh"
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   fi
done

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path \" must be config"
   exit 1
fi
##############执行训练##########
cd $cur_path

#参数修改
sed -i "s|num_batches = int(num_files / batch_size)|num_batches = 10|g" ./SRNTT/model.py
cp -r ${data_path}/SRNTT/ ./SRNTT/models/
cp ${data_path}/imagenet-vgg-verydeep-19.mat ./SRNTT/models/VGG19/

start=$(date +%s)
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    export DEVICE_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d $cur_path/test/output ];then
        rm -rf $cur_path/test/output/${ASCEND_DEVICE_ID}
        mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID/ckpt
    fi

    nohup python3 main.py \
        --is_train True \
        --input_dir ${data_path}/data/train/CUFED/input \
        --ref_dir ${data_path}/data/train/CUFED/ref \
        --map_dir ${data_path}/data/train/CUFED/map_321 \
        --use_pretrained_model False \
        --num_init_epochs 2 \
        --num_epochs 2 \
        --save_dir demo_training_srntt > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait

end=$(date +%s)
e2e_time=$(( $end - $start ))

#参数回改
sed -i "s|num_batches = 10|num_batches = int(num_files / batch_size)|g" ./SRNTT/model.py

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep "perf:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $2}'`
wait
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${TrainingTime}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep "train_acc " $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $12}'|sed 's/,//g'`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "loss:l_rec" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $3}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
echo "TrainAccuracy = None" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log

