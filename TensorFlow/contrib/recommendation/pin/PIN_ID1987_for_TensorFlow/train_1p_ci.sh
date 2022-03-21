#!/bin/bash

echo "`date +%Y%m%d%H%M%S`-[INFO] start to run train_1p_ci.sh "

Network="PIN_ID1987_for_TensorFlow"
#####################
# 【必填】对标性能和精度指标，来源于论文或者GPU复现
benchmark_fps=""
benchmark_accu=""


#####################
# 训练标准化参数列表，请通过参数进行相关路径传递
# 1、数据集路径地址，若不涉及则跳过
npu_data_path=""

# 2、预加载checkpoint地址，若不涉及则跳过
npu_ckpt_path=""

# 3、需要加载的其他文件地址，若不涉及则跳过
npu_other_path=""

# 4、训练输出的地址，若不涉及则跳过
npu_output_path=""  #../log


#####################
# 训练执行拉起命令，打屏信息输出到train_output.log文件
sed -i "s@'./iPinYou-all'@'${npu_data_path}'@g" datasets/iPinYou.py
python3 train.py  | tee train_output.log
sed -i "s@'${npu_data_path}'@./iPinYou-all@g" datasets/iPinYou.py
#####################
# 【选填】基于当前输出的train_output.log计算出NPU性能和精度值
npu_fps="请通过train_output.log输出信息，计算出fps性能值"
npu_accu="请通过train_output.log输出信息，计算出accu性能值"

echo "`date +%Y%m%d%H%M%S`-[INFO] finish to run train_1p_ci.sh "