#! /bin/bash
#Ascend社区已预置的数据集、预训练模型、ATC-OM模型等
#originnal_pic=$ImageNet2012_val
model="/home/HwHiAiUser/AscendProjects/SID/SID.om"
input="/home/HwHiAiUser/AscendProjects/SID/dataset/Sony/short"
output="/home/HwHiAiUser/AscendProjects/SID/out/"
label="ground_truth/val_map.txt"

#开发者个人独立预置的数据集、预训练模型、ATC-OM模型等，支持从OBS仓下载
#obsutil cp obs://obsxxx/xxx/xxx.om ./model/ -f -r

#testcase主体，开发者根据不同模型写作
#preprocess
#python3.7.5 img_preprocess.py --src_path=$originnal_pic --dst_path=$input --pic_num=100
#start inference
./msame --model $model --input $input --output $output 2>&1 |tee inference.log
#top1 accuarcy
python3.7.5 psnr.py $output $label 2>&1 |tee top1.log

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
avg_time=`grep "Inference average time without first time:" inference.log | awk '{print $7}'`
top1=`grep "Top1 accuarcy:" top1.log | awk '{print $7}'`

expect_time=145
expect_top1=0.86
echo "Average inference time is $avg_time ms, expect time is <$expect_time ms"
echo "Top1 accuarcy is $top1, expect top1 is >$expect_top1"
if [[ $avg_time < $expect_time ]] && [[ $top1 > $expect_top1 ]] ;then
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi