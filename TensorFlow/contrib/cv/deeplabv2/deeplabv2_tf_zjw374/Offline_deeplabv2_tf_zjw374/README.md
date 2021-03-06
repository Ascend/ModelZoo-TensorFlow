## 1、原始模型
模型下载来自：由训练脚本产生的模型。桶地址：obs://modelarts-zjw/Offline_deeplabv2/ck_model

将保存的ckpt文件转化为pb文件之后，使用ATC工具转化成om模型

## 2、转om模型
obs链接：obs://modelarts-zjw/Offline_deeplabv2/om_model

ATC转换命令：

/home/HwHiAiUser/Ascend/ascend-toolkit/20.1.rc1/atc/bin/atc --input_shape="input:1,321,321,3" --check_report=/home/HwHiAiUser/modelzoo/deeplabv2-2000/device/network_analysis.report --input_format=NHWC --output="/home/HwHiAiUser/modelzoo/deeplabv2-2000/device/deeplabv2-2000" --soc_version=Ascend310 --framework=3 --model="/home/HwHiAiUser/910model/deeplabv2.pb"  

## 3、代码及路径解释

```
deeplabv2
└─
  ├─README.md
  ├─LICENSE  
  ├─bin_dataset             用于存放验证集.bin文件         桶地址 obs://modelarts-zjw/Offline_deeplabv2/bin_dataset
  ├─output                  用于存放推理后的预测值.bin文件
  ├─data                    用于存放标签文件
  ├─model                   用于存放om模型                桶地址 obs://modelarts-zjw/Offline_deeplabv2/om_model
  ├─ck_model                用于存放checkpoint模型        桶地址 obs://modelarts-zjw/Offline_deeplabv2/ck_model
  ├─pb_model                用于存放pb模型                桶地址 obs://modelarts-zjw/Offline_deeplabv2/pb_model
  ├─model_freeze.py         将.ckpt模型转换为.pb模型
  ├─310img_preprocess.py    将RGB图像转换为bin格式
  ├─310Inference.py         推理
  ├─start_inference.sh      执行推理、验证脚本文件
```


## 4、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具

## 5、数据处理(这里我们直接给出处理后的数据)

从 obs://modelarts-zjw/Offline_deeplabv2/bin_dataset 下载验证集至bin_dataset文件夹


## 6、性能、精度测试：
这个网络是使用deeplabv2来实现对PASCAL VOC 2012 dataset.图像分割，输入为321x321的RGB图像
推理输入的格式：Batch_size:1，shape:1x321x321x3.

bash start_inference.sh

![输入图片说明](https://images.gitee.com/uploads/images/2021/0120/123626_96448e59_8310380.png "屏幕截图.png")
![输入图片说明](https://images.gitee.com/uploads/images/2021/0120/122920_4c9bbf21_8310380.png "屏幕截图.png")

Inference average time without first time: 117.04 ms 推理精度为0.703。