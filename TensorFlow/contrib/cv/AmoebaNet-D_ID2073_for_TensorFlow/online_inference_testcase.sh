#! /bin/bash

#使用ImageNet2012数据集，配置DATA_DIR数据集路径和MODEL_DIR保存checkpoint路径
DATA_DIR=/home/test_user03/tf_records/
MODEL_DIR=/home/test_user03/hh

#开发者个人独立预置的数据集、预训练模型、ATC-OM模型等，支持从OBS仓下载
#obsutil cp obs://obsxxx/xxx/xxx.ckpt ./model/ -f -r

#testcase主体，开发者根据不同模型写作
python3 amoeba_net.py \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --num_cells=6 \
  --image_size=224 \
  --num_epochs=1 \
  --train_batch_size=64 \
  --eval_batch_size=64 \
  --lr=2.56 \
  --lr_decay_value=0.88 \
  --lr_warmup_epochs=0.35 \
  --mode=predict \
  --iterations_per_loop=1251 \
  --num_train_images=10000 \
  --num_eval_images=1000 \
  --eval_timeout=10 \
  > predict.log 2>&1 
#在线推理测试用例只推理1000张图片，并保存打印信息至predict.log

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="Restoring parameters from"  #功能检查字
key2="Inference speed ="  #性能检查字



if [ `grep -c "$key1" "predict.log"` -ne '0' ] && [ `grep -c "$key2" "predict.log"` -ne '0' ];then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi