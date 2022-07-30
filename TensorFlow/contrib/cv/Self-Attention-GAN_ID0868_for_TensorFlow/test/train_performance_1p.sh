#!/bin/bash

# shell脚本所在路径(test目录)
cur_path=`echo $(cd $(dirname $0);pwd)`

# 判断当前shell是否是performance
perf_flag=`echo $0 | grep performance | wc -l`

# 当前执行网络的名称
Network=`echo $(cd $(dirname $0);pwd) | awk -F "/" '{print $(NF-1)}'`

# 路径参数初始化
data_path=""
output_path=""
use_npu=0

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    if [ x"${perf_flag}" = x1 ];then
        echo "usage:./train_performance_1p.sh <args>"
    else
        echo "usage:./train_full_1p.sh <args>"
    fi
    echo " "
    echo "parameter explain:
    --data_path              # dataset of training
    --output_path            # output of training
    --train_steps            # max_step for training
    --train_epochs           # max_epoch for training
    --batch_size             # batch size
    --use_npu                # use_npu
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
    elif [[ $para == --use_npu* ]];then
        use_npu=`echo ${para#*=}`
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
print_log="${output_path}/run.log"
echo "### get your log here : ${print_log}"

CaseName=""
function get_casename()
{
    if [ x"${perf_flag}" = x1 ];then
        CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'perf'
    else
        CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'acc'
    fi
}

# 跳转到code目录
cd ${cur_path}/../
rm -rf ${output_path}/*

# 配置coredump
echo "core-%e-%p-%t" > /proc/sys/kernel/core_pattern
ulimit -c unlimited

##########**********NPU配置信息**********##########
if [[ $use_npu != 0 ]];then
    # 训练任务ID，用户自定义，仅支持大小写字母，数字，中划线，下划线。不建议使用以0开始的纯数字
    export JOB_ID=10087
    # 指定昇腾AI处理器的逻辑ID，单P训练也可不配置，默认为0，在0卡执行训练
    export ASCEND_DEVICE_ID=0
    # 指定训练进程在集合通信进程组中对应的rank标识序号，单P训练固定配置为0
    export RANK_ID=0
    # 指定当前训练进程对应的Device在本集群大小，单P训练固定配置为1
    export RANK_SIZE=1
    # 如果训练脚本中没有使用集合通信接口，可以不配置
    # export RANK_TABLE_FILE=/root/rank_table_1p.json

    # 为了后续方便定位问题，用户也可以通过环境变量使能Dump计算图。
    # 训练任务启动后，会在DUMP_GRAPH_PATH指定的路径下生成若干dump图文件，包括".pbtxt"和".txt"dump文件。
    # 1：全量dump；2：不含有权重等数据的基本版dump；3：只显示节点关系的精简版dump
    # export DUMP_GE_GRAPH=2
    # 把整个流程中各个阶段的图描述信息打印到文件中，此环境变量可以控制dump图的个数。
    # 1：dump所有图；2：dump除子图外的所有图；3：dump最后的生成图
    # export DUMP_GRAPH_LEVEL=2
    # 默认dump图生成在脚本执行目录，可以通过该环境变量指定dump路径
    # export DUMP_GRAPH_PATH=/home/dumpgraph
    # 生成TF_GeOp开头的图数据
    # export PRINT_MODEL=1

    # 配置PROFILING
    # export PROFILING_MODE=true
    # export PROFILING_OPTIONS='{"output":"/tmp/profiling","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization","hccl":"on"}'

    # 设置日志落盘路径
    # Host侧日志路径：$HOME/ascend/log/plog/plog_*.log，$HOME为Host侧用户根目录。
    # Device侧日志路径：$HOME/ascend/log/device-id/device-id_*.log。
    # export ASCEND_PROCESS_LOG_PATH=$HOME/ascend/log

    if [ x"${ASCEND_PROCESS_LOG_PATH}" != x ];then
        rm -rf ${ASCEND_PROCESS_LOG_PATH}
        mkdir -p ${ASCEND_PROCESS_LOG_PATH}
    fi

    # 是否开启日志打屏。开启后，日志将不会保存在log文件中，而是将产生的日志直接打屏显示。
    # export ASCEND_SLOG_PRINT_TO_STDOUT=1
    # 设置应用类日志的全局日志级别及各模块日志级别。
    # 0：对应DEBUG级别；1：对应INFO级别。默认值；2：对应WARNING级别；3：对应ERROR级别；4：对应NULL级别，不输出日志
    # export ASCEND_GLOBAL_LOG_LEVEL=1
    # 设置应用类日志是否开启Event日志。
    # 0：关闭Event日志；1：开启Event日志。默认值
    # export ASCEND_GLOBAL_EVENT_ENABLE=0
    # 业务进程退出前，系统有2000ms的默认延时将Device侧应用类日志回传到Host侧，超时后业务进程退出。
    # export ASCEND_LOG_DEVICE_FLUSH_TIMEOUT=2000
    # 设置"$HOME/ascend/log/plog"和"$HOME/ascend/log/device-id"日志目录下分别能够存储的单个进程的日志文件数量。
    # export ASCEND_HOST_LOG_FILE_NUM=20

    # 获取简单信息
    # export DUMP_GE_GRAPH=3
    # export DUMP_GRAPH_LEVEL=3
    # export PRINT_MODEL=1
    # export ASCEND_GLOBAL_LOG_LEVEL=3
    # export PROFILING_MODE=true
    # export PROFILING_OPTIONS='{"output":"/tmp/profiling","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization","hccl":"on"}'

    # 获取详细信息
    # export DUMP_GE_GRAPH=3
    # export DUMP_GRAPH_LEVEL=1
    # export PRINT_MODEL=1
    # export ASCEND_GLOBAL_LOG_LEVEL=1
    # export PROFILING_MODE=true
    # export PROFILING_OPTIONS='{"output":"/tmp/profiling","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization","hccl":"on"}'
fi
##########**********NPU配置信息**********##########

modelarts_flag=${MODELARTS_MODEL_PATH}
if [ x"${modelarts_flag}" != x ];then
    echo "running with modelarts..."
fi

# 训练开始时间记录，不需要修改
start_time=$(date +%s)

# 基础参数，需要模型审视修改
# 您的训练数据集在${data_path}路径下，请直接使用这个变量获取
# 您的训练输出目录在${output_path}路径下，请直接使用这个变量获取
# 您的其他基础参数，可以自定义增加，但是batch_size请保留，并且设置正确的值

train_epochs=3
batch_size=32
python3.7 main.py --phase train --dataset celebA --gan_type hinge --batch_size=${batch_size} --epoch=${train_epochs} --use_npu=${use_npu} --data_path=${data_path} --output_path=${output_path} 2>&1 | tee ${print_log}

# 性能相关数据计算
StepTime=`grep "sec/step:" ${print_log} | tail -n 10 | awk '{print $10}' | awk '{sum+=$1} END {print sum/NR}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${StepTime}'}'`

# 精度相关数据计算
train_accuracy=`grep "Final accuracy:" ${print_log}  | awk '{print $NF}'`

if [[ $use_npu != 0 ]];then
    # 判断本次执行是否正确使用Ascend NPU
    use_npu_flag=`grep "The model has been compiled on the Ascend AI processor" ${print_log} | wc -l`
    if [ x"${use_npu_flag}" == x0 ];then
        echo "------------------ ERROR NOTICE START ------------------"
        echo "ERROR, your task haven't used Ascend NPU, please check your npu Migration."
        echo "------------------ ERROR NOTICE END------------------"
    else
        echo "------------------ INFO NOTICE START------------------"
        echo "INFO, your task have used Ascend NPU, please check your result."
        echo "------------------ INFO NOTICE END------------------"
    fi

    # 筛选plog中的ERROR日志
    if [ x"${ASCEND_PROCESS_LOG_PATH}" != x ];then
        grep -rh "\[ERROR\]" ${ASCEND_PROCESS_LOG_PATH} > ${output_path}/log_error.txt
        if [ ! -s ${output_path}/log_error.txt ];then
            rm -f ${output_path}/log_error.txt
        fi
    fi
fi

# 获取最终的casename，请保留，case文件名为${CaseName}
get_casename

# 提取所有loss打印信息
grep "step_loss:" ${print_log} | awk '{print $12}' > ${output_path}/${CaseName}_loss.txt

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
ActualLoss=(`awk 'END {print $NF}' $output_path/${CaseName}_loss.txt`)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $output_path/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $output_path/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $output_path/${CaseName}.log
echo "DeviceType = `uname -m`" >> $output_path/${CaseName}.log
echo "CaseName = ${CaseName}" >> $output_path/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $output_path/${CaseName}.log
echo "TrainingTime = ${StepTime}" >> $output_path/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $output_path/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $output_path/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $output_path/${CaseName}.log
