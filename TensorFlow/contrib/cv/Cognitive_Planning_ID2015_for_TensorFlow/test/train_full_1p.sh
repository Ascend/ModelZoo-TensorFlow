#!/bin/bash

##########################################################
#########第3行 至 100行，请一定不要、不要、不要修改##########
#########第3行 至 100行，请一定不要、不要、不要修改##########
#########第3行 至 100行，请一定不要、不要、不要修改##########
##########################################################
# shell脚本所在路径
cur_path=`echo $(cd $(dirname $0);pwd)`

# 判断当前shell是否是performance
perf_flag=`echo $0 | grep performance | wc -l`

# 当前执行网络的名称
Network=`echo $(cd $(dirname $0);pwd) | awk -F"/" '{print $(NF-1)}'`

export RANK_SIZE=1
export RANK_ID=0
export JOB_ID=10087

# 路径参数初始化
data_path=""
output_path=""

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --data_path              # dataset of training
    --output_path            # output of training
    --train_steps            # max_step for training
	  --train_epochs           # max_epoch for training
    --batch_size             # batch size
    -h/--help                show help message
    "
    exit 1
fi

# 参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --output_path* ]];then
        output_path=`echo ${para#*=}`
    elif [[ $para == --train_steps* ]];then
        train_steps=`echo ${para#*=}`
	elif [[ $para == --train_epochs* ]];then
        train_epochs=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be config"
    exit 1
fi

# 校验是否传入output_path,不需要修改
if [[ $output_path == "" ]];then
    output_path="./test/output/${ASCEND_DEVICE_ID}"
fi

# 设置打屏日志文件名，请保留，文件名为${print_log}
print_log="./test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log"
etp_flag=${etp_running_flag}
if [ x"${etp_flag}" != xtrue ];
then
    echo "running without etp..."
    print_log_name=`ls /home/ma-user/modelarts/log/ | grep proc-rank`
    print_log="/home/ma-user/modelarts/log/${print_log_name}"
fi
echo ${print_log}

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

# 训练开始时间记录，不需要修改
start_time=$(date +%s)
##########################################################
#########第3行 至 100行，请一定不要、不要、不要修改##########
#########第3行 至 100行，请一定不要、不要、不要修改##########
#########第3行 至 100行，请一定不要、不要、不要修改##########
##########################################################

#=========================================================
#=========================================================
#========训练执行命令，需要根据您的网络进行修改==============
#=========================================================
#=========================================================
# 您的训练数据集在${data_path}路径下，请直接使用这个变量获取
# 您的训练输出目录在${output_path}路径下，请直接使用这个变量获取
# 您的其他基础参数，可以自定义增加，但是batch_size请保留，并且设置正确的值
batch_size=8

python3 -m pip install --upgrade pip
pip3 install gin-config
pip3 install gym
pip3 install embedders
pip3 install policies
pip3 install tasks

if [ x"${etp_flag}" != xtrue ];
then
    python3.7 ./train_supervised_active_vision.py --mode='train'   --logdir=${output_path}/checkpoint   --modality_types='det'   --batch_size=8   --train_iters=200000   --lstm_cell_size=2048   --policy_fc_size=2048   --sequence_length=20   --max_eval_episode_length=100   --test_iters=194   --gin_config=envs/configs/active_vision_config.gin   --gin_params="ActiveVisionDatasetEnv.dataset_root='${data_path}'"   --logtostderr
else
    echo -------123456-------
    python3.7 ./train_supervised_active_vision.py --mode='train'   --logdir=${output_path}/checkpoint   --modality_types='det'   --batch_size=8   --train_iters=200000   --lstm_cell_size=2048   --policy_fc_size=2048   --sequence_length=20   --max_eval_episode_length=100   --test_iters=194   --gin_config=envs/configs/active_vision_config.gin   --gin_params="ActiveVisionDatasetEnv.dataset_root='${data_path}'"   --logtostderr  > ${print_log}
fi

# 性能相关数据计算
StepTime=`grep "sec/step" ${print_log} | tail -n 10 | awk -F "(" '{print $2}' | awk -F " " '{print $1}' |  awk '{sum+=$1} END {print sum/NR}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${StepTime}'}'`

# 精度相关数据计算
train_accuracy=`grep "loss" ${print_log}   | tail -n 10 | awk -F "=" '{print $2}' | awk -F "(" '{print $1}'`
# 提取所有loss打印信息
grep "loss" ${print_log} | awk -F "=" '{print $2}' | awk -F "(" '{print $1}' > ./test/output/${ASCEND_DEVICE_ID}/my_output_loss.txt


###########################################################
#########后面的所有内容请不要修改###########################
#########后面的所有内容请不要修改###########################
#########后面的所有内容请不要修改###########################
###########################################################

# 判断本次执行是否正确使用Ascend NPU
use_npu_flag=`grep "The model has been compiled on the Ascend AI processor" ${print_log} | wc -l`
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
if [ -f ./test/output/${ASCEND_DEVICE_ID}/my_output_loss.txt ];
then
    mv ./test/output/${ASCEND_DEVICE_ID}/my_output_loss.txt ./test/output/${ASCEND_DEVICE_ID}/${CaseName}_loss.txt
fi

# 训练端到端耗时
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

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