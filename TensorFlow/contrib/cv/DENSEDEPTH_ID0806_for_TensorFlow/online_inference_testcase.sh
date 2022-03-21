#! /bin/bash
#开发者个人独立预置的数据集、预训练模型、ATC-OM模型等，支持从OBS仓下载
obsutil cp obs://bothdata/code_npu_4/output/model/result/1637127130/models/1637125790-n50-e5-bs4-lr0.0001-densedepth_nyu/ckpt_npu/epoch_end/ ./dataset/ -f -r
obsutil cp obs://bothdata/code_npu_4/dataset/nyu/test/nyu_test.zip ./dataset/ -f -r

pip3 install -r requirements.txt

#testcase主体，开发者根据不同模型写作
nohup python3.7  evaluate.py --ckptdir .dataset/ckpt_npu --test_data ./dataset/nyu_test.zip	--logdir ./ --bs 2> inference.log 2>&1

wait

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="log10"

if [ `grep -c "$key1" "inference.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi