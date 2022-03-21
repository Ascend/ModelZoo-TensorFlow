#!/bin/bash

echo "`date +%Y%m%d%H%M%S`-[INFO] start to run train_1p.sh "
#####################
# 【必填】对标性能和精度指标，来源于论文或者GPU复现
benchmark_fps="4.098  steps/sec "
benchmark_accu="AMT “real vs fake” test,labeled by Turkers. 26.8% True"
#####################
# 训练标准化参数列表，请通过参数进行相关路径传递
cur_path=`pwd`
# 1、数据集路径地址，若不涉及则跳过
data_path=''
# 2、预加载checkpoint地址，若不涉及则跳过
ckpt_path=''
# 3、需要加载的其他文件地址，若不涉及则跳过
npu_other_path=''
# 4、训练输出的地址，若不涉及则跳过
output_path="../result/"
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`

    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    fi
done
#####################
# 训练执行拉起命令，打屏信息输出到train_output.log文件
cd $cur_path/
python3 train.py --result=./result \
	--dataset=$data_path \
	--chip=npu \
	--platform=apulis \
	--train_epochs=1 \  | tee train_output.log

#####################
# 【选填】基于当前输出的train_output.log计算出NPU性能和精度值
#npu_fps="请通过train_output.log输出信息，计算出fps性能值"
#npu_accu="请通过train_output.log输出信息，计算出accu性能值"

echo "`date +%Y%m%d%H%M%S`-[INFO] finish to run train_1p.sh "
