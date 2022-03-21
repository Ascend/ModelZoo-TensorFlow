#!/bin/bash
cur_path=`pwd`/../
#export ASCEND_DEVICE_ID=3

#基础参数，需要模型审视修改
#Batch Size
batch_size=8
#网络名称，同目录名称
Network="OSMN_ID1103_for_TensorFlow"
#Device数量，单卡默认为1
RankSize=1
export RANK_SIZE=1
#训练epoch，可选
train_epochs=
#训练step
train_steps=500
#学习率
learning_rate=

#参数配置
data_path="/data"

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh --data_path=../data"
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
##############拷贝数据集########
rm -rf ${cur_path}/models/*
cp -r ${data_path}/models/vgg_16.ckpt ${cur_path}/models/
if [ -f "${data_path}/data/train_annos.pkl" ];then
  if [ -d $cur_path/cache ];then
    rm -rf $cur_path/cache
	mkdir $cur_path/cache
  else
    mkdir $cur_path/cache
  fi
  cp -r ${data_path}/data/*.pkl ${cur_path}/cache/
fi
wait
##############执行训练##########
wait
cd $cur_path
if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
nohup python3 osmn_coco_pretrain.py --data_path $data_path/data/ \
        --model_save_path $cur_path/models/ \
        --training_iters $train_steps \
        --display_iters 100 \
        --save_iters 500 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
tmp_TrainingTime=`grep "time/step" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "time/step = " '{print $2}'|awk '{sum+=$1} END {print sum/NR}'`
TrainingTime=`awk 'BEGIN {printf "%.2f\n", '${tmp_TrainingTime}'}'`
FPS=`awk 'BEGIN {printf "%.2f\n", '${batch_size}'/'${TrainingTime}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"

###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Training Loss = " $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "Training Loss = " '{print $2}' | awk -F " " '{print $1}'> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=(`awk 'END {print $NF}' $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_loss.txt`)

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log