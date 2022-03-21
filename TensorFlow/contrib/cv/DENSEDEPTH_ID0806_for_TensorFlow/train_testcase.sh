#! /bin/bash
#开发者个人独立预置的数据集、预训练模型、ATC-OM模型等，支持从OBS仓下载
obsutil cp obs://bothdata/code_npu_4/dataset/nyu/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5 ./dataset/ -f -r
obsutil cp obs://bothdata/code_npu_4/dataset/nyu/test/nyu_data.tfrecords ./dataset/ -f -r
obsutil cp obs://bothdata/code_npu_4/dataset/nyu/test/nyu_test.zip ./dataset/ -f -r

pip3 install -r requirements.txt

#testcase主体，开发者根据不同模型写作
batch_size=4

epochs=5
steps=500
test_data="./dataset/nyu_test.zip"
train_tfrecords="./dataset/nyu_data.tfrecords"

is_distributed='False'
is_loss_scale='True'
hcom_parallel='False'

op_select_implmode='high_precision'
precision_mode='allow_mix_precision'


nohup python3.7  train.py \
          --test_data="${test_data}" \
          --bs="${batch_size}" \
          --train_tfrecords="${train_tfrecords}" \
          --epochs="${epochs}" \
          --steps="${steps}" \
          --full \
          --op_select_implmode="${op_select_implmode}" \
          --precision_mode="${precision_mode}" \
          --hcom_parallel="${hcom_parallel}" \
          --is_distributed="${is_distributed}" \
          --is_loss_scale="${is_loss_scale}"> train.log 2>&1

wait

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="Train success" #功能检查字
key2="images/sec" #性能检查字
key3="log10" #精度检查字

if [ `grep -c "$key1" "tran.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi