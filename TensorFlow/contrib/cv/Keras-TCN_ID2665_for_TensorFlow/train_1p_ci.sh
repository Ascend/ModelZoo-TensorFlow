#!/bin/bash

echo "`date +%Y%m%d%H%M%S`-[INFO] start to run train_1p.sh "
export JOB_ID=10086
export RANK_SIZE=1
#####################
# ������Ա����ܺ;���ָ�꣬��Դ�����Ļ���GPU����
benchmark_fps=""
benchmark_accu=""
#####################
# ѵ����׼�������б���ͨ�������������·������
cur_path=`pwd`
# 1�����ݼ�·����ַ�������漰������
data_path=''
# 2��Ԥ����checkpoint��ַ�������漰������
ckpt_path=''
# 3����Ҫ���ص������ļ���ַ�������漰������
npu_other_path=''
# 4��ѵ������ĵ�ַ�������漰������
output_path=""
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`

    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    fi
done
#####################
# ѵ��ִ���������������Ϣ�����train_output.log�ļ�
cd $cur_path/
python3 train.py --epochs=1 | tee train_output.log

#####################
# ��ѡ����ڵ�ǰ�����train_output.log�����NPU���ܺ;���ֵ
#npu_fps="��ͨ��train_output.log�����Ϣ�������fps����ֵ"
#npu_accu="��ͨ��train_output.log�����Ϣ�������accu����ֵ"

echo "`date +%Y%m%d%H%M%S`-[INFO] finish to run train_1p.sh "