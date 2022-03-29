#! /bin/bash
#Ascend社区已预置的数据集、预训练模型、ATC-OM模型等
DATA_PATH=./data/
VOCAB_SOURCE=${DATA_PATH}/vocab.share
VOCAB_TARGET=${DATA_PATH}/vocab.share
TRAIN_FILES=${DATA_PATH}/concat128/train.l128.tfrecord-001-of-016

#开发者个人独立预置的数据集、预训练模型、ATC-OM模型等，支持从OBS仓下载
obsutil cp obs://bigelow/dataset4/BSD.tfrecords ./data/ -f -r

code_dir=./
work_dir=./
dataset_path=./data/
output_path=./result/


python3.7 ./main.py --output_dir=${output_path} --phase=train --training_set=${dataset_path}/BSD.tfrecords --batch_size=1 --training_steps=1000 --summary_steps=50 --checkpoint_steps=100 --save_steps=50
if [ $? -eq 0 ];
then
    echo "[CANN-Modelzoo] train return success"
else
    echo "[CANN-Modelzoo] train return failed"
fi

######训练后把需要备份的内容保存到output_path######
###此处每个网络不同，视情况添加cp
cp -r ${work_dir} ${output_path}

######训练后输出目录文件确认######
echo "[CANN-Modelzoo] after train - list my work files[${work_dir}]:"
ls -al ${work_dir}
echo ""

echo "[CANN-Modelzoo] after train - list my output files[${output_path}]:"
ls -al ${output_path}
echo ""