#Faster-RCNN离线推理

Faster-RCNN目标检测模型离线推理

模型论文：https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf

训练代码参考：https://github.com/you359/Keras-FasterRCNN

##环境

Python版本：3.7.5

CANN版本：5.0.3

推理芯片：昇腾310

第三方库：参考requirements.txt

##快速入门

###1.数据集

使用VOC2012数据集，下载链接：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

###2.数据预处理

解压数据集：

```
python3 ./scripts/unzip.py
```

对数据集中的图片进行resize，便于统一图片尺寸进行推理：

```
python3 ./scripts/resize_image.py
```

将VOC2012数据集的验证集部分转换为二进制文件：

```
mkdir ./input_bins_x600
mkdir ./input_bins_x800
python3 ./scripts/convert_binary.py
```

###3.配置环境

```
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}
```

###4.将pb文件转换为om模型

```
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=/root/FasterRCNN/FasterRCNN-PR/pb_model/rpn.pb --framework=3 --output=/root/FasterRCNN/FasterRCNN-PR/pb_model/rpn_x800 --output_type=FP32 --soc_version=Ascend310 \
        --input_shape="Input:1,800,600,3" \
        --precision_mode=force_fp32\
        --log=info \
        --out_nodes="Identity:0;Identity_1:0;Identity_2:0"
```

```
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=/root/FasterRCNN/FasterRCNN-PR/pb_model/classifier.pb --framework=3 --output=/root/FasterRCNN/FasterRCNN-PR/pb_model/classifier_x800 --output_type=FP32 --soc_version=Ascend310 \
        --input_shape="Input:1,50,38,1024;Input_1:1,150,4" \
        --precision_mode=force_fp32\
        --log=info \
        --out_nodes="Identity:0;Identity_1:0"
```

```
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=/root/FasterRCNN/FasterRCNN-PR/pb_model/rpn.pb --framework=3 --output=/root/FasterRCNN/FasterRCNN-PR/pb_model/rpn_x600 --output_type=FP32 --soc_version=Ascend310 \
        --input_shape="Input:1,600,800,3" \
        --precision_mode=force_fp32\
        --log=info \
        --out_nodes="Identity:0;Identity_1:0;Identity_2:0"
```

```
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=/root/FasterRCNN/FasterRCNN-PR/pb_model/classifier.pb --framework=3 --output=/root/FasterRCNN/FasterRCNN-PR/pb_model/classifier_x600 --output_type=FP32 --soc_version=Ascend310 \
        --input_shape="Input:1,38,50,1024;Input_1:1,150,4" \
        --precision_mode=force_fp32\
        --log=info \
        --out_nodes="Identity:0;Identity_1:0"
```

pb和om模型的链接：

https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/Faster-RCNN/classifier.pb       
https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/Faster-RCNN/classifier_x800.om  
https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/Faster-RCNN/rpn_x600.om
https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/Faster-RCNN/classifier_x600.om  
https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/Faster-RCNN/rpn.pb              
https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/Faster-RCNN/rpn_x800.om

###5.构建推理程序

```
bash build.sh
```

###6.进行Faster-RCNN第一阶段的推理

进行rpn部分的推理，注意修改benchmark_tf_rpn.sh中的相应路径：

```
bash ./scripts/benchmark_tf_rpn.sh
```

处理第一阶段的推理结果，作为第二阶段推理的输入：

```
mkdir ./input_classifier_x800_0
mkdir ./input_classifier_x800_1
mkdir ./input_classifier_x600_0
mkdir ./input_classifier_x600_1
python3 ./scripts/input_classifier.py
```

###7.进行Faster-RCNN第二阶段的推理

进行Faster-RCNN第二阶段的推理，注意修改benchmark_tf_classfier.sh中的相应路径：

```
bash ./scripts/benchmark_tf_classfier.sh
```

###8.推理结果的后处理与mAP的获取

```
mkdir ./map_out
mkdir ./map_out/detection-results
mkdir ./map_out/ground-truth
python3 ./scripts/post_processing.py
python3 ./scripts/get_map.py
```

##推理结果

VOC2012验证集：mAP = 77.14%，推理结果的更多细节保存在results目录下。

