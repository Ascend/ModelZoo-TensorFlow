#!/bin/bash

cur_path=`pwd`/../
export ENABLE_RUNTIME_V2=1

#失败用例日志打印
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

#基础参数，需要模型审视修改
#Batch Size
batch_size=64
#网络名称，同目录名称
Network="Word2vec_ID2350_for_TensorFlow2.X"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=1
#训练step
train_steps=3000
#学习率
learning_rate=0.001

#参数配置
data_path=""
############维测参数##############
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
if [[ $over_dump == True ]];then
    over_dump_path=$cur_path/test/output/overflow_dump #此处cur_path为代码根目录
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
if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh $data_path --work_dir="$cur_path/estimator_working_dir" --export_path="$cur_path/outputs/models/000001-first_generation""
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
    elif [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
        source set_conda.sh
        echo "conda_name: $conda_name"
        source activate $conda_name
    fi
done

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh"
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
wait

if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

mkdir -p $cur_path/logs
cp -r ${data_path}/model ${cur_path}

start=$(date +%s)
nohup python3 main.py --epochs=${train_epochs} \
                        --data_path=${data_path} \
                        --static=True \
                        --model_path=${cur_path}/model/w2v_model/ \
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
    --auto_tune=${auto_tune} \
    --profiling_dump_path=${profiling_dump_path} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait

end=$(date +%s)
e2e_time=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
#FPS=`grep "trainloss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v INFO|awk '{print $NF}'|tail -n +2|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
count_time=`grep Perf $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | tail -n 50 | awk 'NR==1' | awk '{print $(NF)}'`
final_time=`grep Perf $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | tail -n 1 | awk 'NR==1' | awk '{print $(NF)}'`
total_time=`awk 'BEGIN{printf "%.5f\n",'${final_time}'-'${count_time}'}'`
TrainingTime=`awk 'BEGIN{printf "%.5f\n",'${total_time}'/'50'}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${TrainingTime}'}'`
wait
#TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${FPS}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#吞吐量
ActualFPS=${FPS}

###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}_bs${BatchSize}_${RankSize}'p_RT2_perf'

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视

cat $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |grep "training batch"|awk '{print $5}'|tr -d ',' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#打印，不需要修改

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

train_accuracy=`grep "overall best f1" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $5}'`

##获取错误信息
#系统错误信息
#error_msg="ModuleNotFoundError: No module named 'impl.conv2d_backprop_filter'"
#判断错误信息是否和历史状态一致，此处无需修改
#Status=`grep "${error_msg}" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | wc -l`
#失败阶段，枚举值图准备FAIL/图拆分FAIL/图优化FAIL/图编译FAIL/图执行FAIL/流程OK
#ModelStatus="图执行FAIL"
#DTS单号或者issue链接
#DTS_Number="TuscanyV1R1C80_IR_0004_SF_220_SR_010_AR_001"

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "ModelStatus = ${ModelStatus}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "DTS_Number = ${DTS_Number}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "Status = ${Status}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "error_msg = ${error_msg}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log