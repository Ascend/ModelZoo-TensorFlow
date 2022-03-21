#! /bin/bash
#开发者个人独立预置的数据集、预训练模型、ATC-OM模型等，支持从OBS仓下载
obsutil cp obs://bothdata/n2n/datasets/network_final.pickle ./datasets/ -f -r
obsutil cp obs://bothdata/n2n/datasets/kodak/ ./datasets/ -f -r

pip3 install -r requirements.txt

#testcase主体，开发者根据不同模型写作
nohup python3.7.5 config.py \
        --graph-run-mode=0 \
        --op-select-implmode=high_precision \
        --precision-mode=allow_mix_precision \
        validate \
        --dataset-dir=datasets/kodak \
        --network-snapshot=datasets/network_final.pickle > inference.log 2>&1 &

wait

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key3="Average PSNR" #精度检查字

if [ `grep -c "$key3" "inference.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi