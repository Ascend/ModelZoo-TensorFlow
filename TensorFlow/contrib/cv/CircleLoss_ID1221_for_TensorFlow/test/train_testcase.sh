#! /bin/bash
#开发者个人独立预置的数据集、预训练模型、ATC-OM模型等，支持从OBS仓下载
obsutil cp obs://liuyixin2/datasets/Market-1501-v15.09.15/resnet_v1_50.ckpt ./dataset/ -f -r
obsutil cp obs://liuyixin2/datasets/Market-1501-v15.09.15/bounding_box_train.zip ./dataset/ -f -r
unzip -d ./dataset/ ./dataset/bounding_box_train.zip
pip3 install -r requirements.txt


#testcase主体，开发者根据不同模型写作
batch_size=16
epochs=1
data_url="./dataset/"
train_url="./dataset/"

nohup python3.7  train_npuv3.py \
          --data_url="${data_url}" \
          --epoch_size="${epochs}" \
          --train_url="${train_url}" > train.log 2>&1

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