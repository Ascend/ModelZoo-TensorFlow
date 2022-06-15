#! /bin/bash

#使用FSNS数据集，配置dataset_dir数据集路径和train_log_dir保存checkpoint路径以及checkpoint_inception初试权重路径
train_log_dir=./fsns/
train_log_dir=./ckpt/
checkpoint_inception=./inception_v3.ckpt

#开发者个人独立预置的数据集、预训练模型、ATC-OM模型等，支持从OBS仓下载
obsutil cp obs://cann-2021-10-21/attention_ocr/attention_ocr/python/datasets/data/fsns/ ./fsns/ -f -r
obsutil cp obs://cann-2021-10-21/attention_ocr/attention_ocr/python/inception_v3.ckpt   ./      -f -r

#testcase主体，开发者根据不同模型写作
python3 NPU_train.py \
  --dataset_dir=${data_path} \
  --train_log_dir=${output_path} \
  --checkpoint_inception=${checkpoint_inception} \
  --max_number_of_steps=200 \
  --log_interval_steps=200 \
  > train.log 2>&1 
#训练测试用例只训练200个step，并保存打印信息至train.log

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="Saving checkpoints"  #功能检查字
key2="sec/step :"  #性能检查字
key3="loss ="  #精度检查字


if [ `grep -c "$key1" "train.log"` -ne '0' ] && [ `grep -c "$key2" "train.log"` -ne '0' ] && [ `grep -c "$key3" "train.log"` -ne '0' ];then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi