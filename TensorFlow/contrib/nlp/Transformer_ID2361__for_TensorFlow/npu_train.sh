#!/bin/bash

code_dir=$1
work_dir=$2
dataset_path=$3
output_path=$4

#############训练前输入目录文件确认#########################
echo "[CANN-ZhongZhi] before train - list my run files[/usr/local/Ascend/ascend-toolkit]:"
ls -al /usr/local/Ascend/ascend-toolkit
echo ""

echo "[CANN-ZhongZhi] before train - list my code files[${code_dir}]:"
ls -al ${code_dir}
echo ""

echo "[CANN-ZhongZhi] before train - list my work files[${work_dir}]:"
ls -al ${work_dir}
echo ""

echo "[CANN-ZhongZhi] before train - list my dataset files[${dataset_path}]:"
ls -al ${dataset_path}
echo ""

echo "[CANN-ZhongZhi] before train - list my output files[${output_path}]:"
ls -al ${output_path}
echo ""

######环境变量修改######
###如果需要修改环境变量的，在此处修改
###搭配最大内存使用
#echo "GE_USE_STATIC_MEMORY ${GE_USE_STATIC_MEMORY}"
#echo $GE_USE_STATIC_MEMORY
#echo "GE_USE_STATIC_MEMORY"
#export GE_USE_STATIC_MEMORY=1
#echo "GE_USE_STATIC_MEMORY ${GE_USE_STATIC_MEMORY}"
#echo $GE_USE_STATIC_MEMORY
#echo "GE_USE_STATIC_MEMORY"


##接口老哥提示打开
echo "ENABLE_FORCE_V2_CONTROL ${GE_USE_STATIC_MEMORY}"
export ENABLE_FORCE_V2_CONTROL=1
echo "ENABLE_FORCE_V2_CONTROL ${GE_USE_STATIC_MEMORY}"
#设置日志级别为info
#export ASCEND_GLOBAL_LOG_LEVEL=1
#设置日志打屏到屏幕
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export TF_CPP_MIN_LOG_LEVEL=0
env > ${output_path}/my_env.log

######训练执行######
###此处每个网络执行命令不同，需要修改
python3.7 ${code_dir}/train621V.py --data_url=${dataset_path} --train_url=${output_path}
if [ $? -eq 0 ];
then
    echo "[CANN-ZhongZhi] train return success"
else
    echo "[CANN-ZhongZhi] train return failed"
fi

######训练后把需要备份的内容保存到output_path######
###此处每个网络不同，视情况添加cp
cp -r ${work_dir} ${output_path}

######训练后输出目录文件确认######
echo "[CANN-ZhongZhi] after train - list my work files[${work_dir}]:"
ls -al ${work_dir}
echo ""

echo "[CANN-ZhongZhi] after train - list my output files[${output_path}]:"
ls -al ${output_path}
echo ""
