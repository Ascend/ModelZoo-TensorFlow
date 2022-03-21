#! /bin/bash
#Ascend社区已预置的数据集、预训练模型、ATC-OM模型等

#开发者个人独立预置的数据集、预训练模型、ATC-OM模型等，支持从OBS仓下载
obsutil cp obs://stnet-id2360/npu/MA-new-STNet_ID2360_for_TensorFlow-01-20-14-07/output/stnet_om.om ./ 
obsutil cp obs://stnet-id2360/dataset/mnist_sequence1_sample_5distortions5x5.npz ./dataset/data.npz
#testcase主体，开发者根据不同模型写作
#preprocess
python3 preprocess.py --total_num 100 --data_path ./dataset/data.npz
python img2bin.py -i ./images -t float32 -o ./out -w 40 -h 40 -f GRAY -c [0.00392]
#start inference
python3 stnet_offline_inference.py --model_path ./stnet_om.om --output_path ./out --data_path ./dataset/data.npz --inference_path ./inference_out/ &>inference.log

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
avg_time=`grep "Inference average time without first time:" inference.log | awk '{print $7}'`
accuracy=`grep "Final Offline Inference Accuracy:" inference.log | awk '{print $5}'`

expect_time=3.0
expect_accuracy=0.93
echo "Average inference time is $avg_time ms, expect time is $expect_time ms"
echo "Top1 accuarcy is $accuracy, expect top1 is $expect_accuracy"
if [[ $avg_time < $expect_time ]] && [[ $accuracy > $expect_accuracy ]] ;then
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi