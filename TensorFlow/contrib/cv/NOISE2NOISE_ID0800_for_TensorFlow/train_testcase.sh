#! /bin/bash
#开发者个人独立预置的数据集、预训练模型、ATC-OM模型等，支持从OBS仓下载
obsutil cp obs://bothdata/n2n/datasets/bsd300.tfrecords ./datasets/ -f -r
obsutil cp obs://bothdata/n2n/datasets/kodak/ ./datasets/ -f -r

pip3 install -r requirements.txt

#testcase主体，开发者根据不同模型写作
noise='gaussian'
noise2noise='True'
long_train='False'
train_tfrecords='datasets/bsd300.tfrecords'

is_distributed='False'
is_loss_scale='True'
hcom_parallel='False'

graph_run_mode=1 # train
op_select_implmode='high_precision'
precision_mode='allow_mix_precision'


nohup python3.7 config.py --graph-run-mode="${graph_run_mode}" \
    --op-select-implmode="${op_select_implmode}" \
    --precision-mode="${precision_mode}" \
    train \
    --noise="${noise}" \
    --noise2noise="${noise2noise}" \
    --long-train="${long_train}" \
    --train-tfrecords="${train_tfrecords}" \
    --hcom-parallel="${hcom_parallel}" \
    --is-distributed="${is_distributed}" \
    --is-loss-scale="${is_loss_scale}"  > tran.log 2>&1 &

wait

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="Finished train.train()" #功能检查字
key2="sec/iter" #性能检查字
key3="Average PSNR" #精度检查字

if [ `grep -c "$key3" "tran.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi