#!/bin/bash

# shell脚本所在路径
cur_path=`echo $(cd $(dirname $0);pwd)`

# 判断当前shell是否是performance
perf_flag=`echo $0 | grep performance | wc -l`
# 当前执行网络的名称
Network="GMH—MDN_ID1225_for_TensorFlow"
#失败用例打屏
export ASCEND_SLOG_PRINT_TO_STDOUT=1
#基础参数，需要模型审视修改
#batch Size
batch_size=64
#当前是否为测试，默认为False，即训练模式
test="False"
#网络名称，同目录名称
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
epochs=1
#学习率
learning_rate='1e-3'
#参数配置
cameras_path="../data/h36m/cameras.h5"
data_dir="../data/h36m/"
train_dir="../experiments/test_git/"
load_dir="../Models/mdm_5_prior/"
load=0

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh <arg>"

   echo ""
   echo "parameter explain:
    --test          Set to True for sampling
    --learning_rate	Learning rate
    --batch_size	batch size to use during training
    --epochs		How many epochs we should train for
    --cameras_path	Directory to load camera parameters
    --data_dir		Data directory
    --train_dir		Training directory
    --load_dir		Specify the directory to load trained model
    --load		    Try to load a previous checkpoint
    -h/--help       Show help message
   "
   exit 1
fi

# 参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_dir* ]];then
        data_dir=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    elif [[ $para == --train_dir* ]];then
        train_dir=`echo ${para#*=}`
    elif [[ $para == --load_dir* ]];then
        load_dir=`echo ${para#*=}`
    elif [[ $para == --cameras_path* ]];then
        cameras_path=`echo ${para#*=}`
    elif [[ $para == --load* ]];then
        load=`echo ${para#*=}`
    elif [[ $para == --learning_rate* ]];then
        learning_rate=`echo ${para#*=}`
    elif [[ $para == --test* ]];then
        test=`echo ${para#*=}`
    fi
done

# 校验是否传入data_dir,不需要修改
if [[ $data_dir  == "" ]];then
   echo "[Error] para \"data_dir\" must be config"
   exit 1
fi

# 校验是否传入train_dir,不需要修改
if [[ $train_dir  == "" ]];then
   echo "[Error] para \"train_dir\" must be config"
   exit 1
fi

# 设置打屏日志文件名，请保留，文件名为${print_log}
print_log="./test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log"
modelarts_flag=${MODELARTS_MODEL_PATH}
if [ x"${modelarts_flag}" != x ];
then
    echo "running without etp..."
    print_log_name=`ls /home/ma-user/modelarts/log/ | grep proc-rank`
    print_log="/home/ma-user/modelarts/log/${print_log_name}"
fi
echo "### get your log here : ${print_log}"

CaseName=""
function get_casename()
{
    if [ x"${perf_flag}" = x1 ];
    then
        CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'perf'
    else
        CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'acc'
    fi
}

# 跳转到code目录
cd ${cur_path}/../
rm -rf ./test/output/${ASCEND_DEVICE_ID}
mkdir -p ./test/output/${ASCEND_DEVICE_ID}
touch ./test/output/${ASCEND_DEVICE_ID}/my_output_loss.txt
cd ${cur_path}/../src

start=$(date +%s)

python3 ./predict_3dpose_mdm.py \
--data_dir ${data_dir} \
--train_dir ${train_dir} > ${print_log}

#python3 ./predict_3dpose_mdm.py \
#--data_dir ${data_dir} \
#--train_dir ${train_dir} \
#--test ${test} \
#--load_dir ${load_dir} \
#--cameras_path ${cameras_path} \
#--load ${load}
#--batch_size ${batch_size} \
#--epochs ${epochs} \
#--learning_rate ${learning_rate} > ${print_log}
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))

#输出性能FPS，需要模型审视修改
StepTime=`grep "done in" ${print_log} | tail -n 10|awk '{print $3}' | awk '{sum+=$1} END {print sum/NR/1000}'`
#打印，不需要修改
FPS=`awk 'BEGIN{printf("%.2f\n", '${batch_size}' /'${StepTime}')}'`

#输出训练精度,需要模型审视修改
train_accuracy=`grep "root - Average" ${print_log} | awk 'END {print $7}'`

# 提取所有loss打印信息
grep "Train loss avg:" ${print_log} | awk '{print $4}' > $cur_path/output/${ASCEND_DEVICE_ID}/my_output_loss.txt

# 判断本次执行是否正确使用Ascend NPU
use_npu_flag=`grep "tf_adapter" ${print_log} | wc -l`
if [ x"${use_npu_flag}" == x0 ];
then
    echo "------------------ ERROR NOTICE START ------------------"
    echo "ERROR, your task haven't used Ascend NPU, please check your npu Migration."
    echo "------------------ ERROR NOTICE END------------------"
else
    echo "------------------ INFO NOTICE START------------------"
    echo "INFO, your task have used Ascend NPU, please check your result."
    echo "------------------ INFO NOTICE END------------------"
fi

# 获取最终的casename，请保留，case文件名为${CaseName}
get_casename

# 重命名loss文件
if [ -f $cur_path/output/${ASCEND_DEVICE_ID}/my_output_loss.txt ];
then
    mv $cur_path/output/${ASCEND_DEVICE_ID}/my_output_loss.txt $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}_loss.txt
fi

echo "------------------ Final result ------------------"
# 输出性能FPS/单step耗时/端到端耗时
echo "Final Performance images/sec : $FPS"
echo "Final Performance sec/step : $StepTime"
echo "E2E Training Duration sec : $e2e_time"

# 输出训练精度
echo "Final Train Accuracy : ${train_accuracy}"

# 最后一个迭代loss值，不需要修改
ActualLoss=(`awk 'END {print $NF}' $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}_loss.txt`)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = `uname -m`" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${StepTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log