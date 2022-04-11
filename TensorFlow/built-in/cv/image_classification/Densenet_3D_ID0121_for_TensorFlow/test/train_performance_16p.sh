#!/bin/bash


cur_path=`pwd`/..
RANK_ID_START=0
export RANK_ID=0
export RANK_SIZE=16
export JOB_ID=888886

# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数 需要模型审视修改
#网络名称，同目录名称
Network="Densenet_3D_ID0121_for_TensorFlow"
batch_size=2
#维测参数，precision_mode需要模型审视修改
autotune=False

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --autotune* ]];then
        autotune=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
        elif [[ $para == --server_index* ]];then
        server_index=`echo ${para#*=}`
    elif [[ $para == --conf_path* ]];then
        conf_path=`echo ${para#*=}`
    fi
done

# 自动生成ranktable的脚本
rank_size=8
nohup python3 set_ranktable.py --npu_nums=$((RANK_SIZE/rank_size)) --conf_path=$conf_path
wait
export RANK_TABLE_FILE=${cur_path}/test/rank_table.json

start=$(date +%s)
for((RANK_ID=$((rank_size*server_index));RANK_ID<$((((server_index+1))*rank_size));RANK_ID++));
do
    # 设置环境变量
    export RANK_ID=$RANK_ID
    export DEVICE_INDEX=`expr ${RANK_ID} - $((rank_size*server_index))`
    export ASCEND_DEVICE_ID=`expr ${RANK_ID} - $((rank_size*server_index))`
    ASCEND_DEVICE_ID=`expr ${RANK_ID} - $((rank_size*server_index))`
    echo "DEVICE ID: $ASCEND_DEVICE_ID"
    #进入训练脚本目录，需要模型审视修改
    if [ -d $cur_path/test/output/$ASCEND_DEVICE_ID ];then
       rm -rf $cur_path/test/output/$ASCEND_DEVICE_ID
       mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
    else
       mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
    fi
    cd $cur_path
    python3 train.py -bs 2 -mn dense24 -sp dense24_correction -nc True -e 1 -r ${data_path} -per True -mul_rank_size=$RANK_SIZE -mul_device_id=$RANK_ID > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait
end=$(date +%s)
e2etime=$(( $end - $start ))
step_sec=`grep -a 'epoch-patient' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log |awk 'END {print $16}'`
ActualFPS=`awk 'BEGIN{printf "%.2f\n",'${RANK_SIZE}'*'${batch_size}'/'$step_sec'}'`
echo "--------Final Result ----------"
echo "Final Performance ms/step : $ActualFPS"
echo "Final Training Duration sec : $e2etime"
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
grep 'patient acc:' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log|awk '{print $6}'|sed 's/,//g' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_acc.txt
#最后一个迭代acc值，不需要修改
train_accuracy=`awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_acc.txt`

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
#稳定性精度看护结果汇总
#训练用例信息，不需要修改
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据
#单迭代训练时长，不需要修改
TrainingTime=`grep "time cust:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $16}'`
#ActualFPS=`echo "scale=2;${BatchSize} / ${TrainingTime}"|bc`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'patient loss:' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log|awk '{print $3}'|sed 's/,//g' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
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
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log