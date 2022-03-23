## 模型功能

 对图像中的棋子类别进行分类。

## 原始模型

参考实现 ：
https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/built-in/TensorFlow/Official/cv/image_classification/VGG16_for_TensorFlow


原始pb模型网络下载地址 ：https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/robot_play_chess/chess_ckpt_0804_vgg_99.pb



AIPP下载地址 ：https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/robot_play_chess/insert_op.cfg




使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
atc --output_type=FP32 --input_shape="vgg16_input:1,56,56,3"  --input_format=NHWC --output="chess_ckpt_0804_vgg_99" --soc_version=Ascend310 --insert_op_conf=insert_op.cfg --framework=3 --model="./chess_ckpt_0804_vgg_99.pb"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/chess_ckpt_0804_vgg_99.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 10

```

```
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model /home/HwHiAiUser/l00612380/level3_multi_model/Robot_Play_Chess-master/models/chess_ckpt_0804_vgg_99.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
/home/HwHiAiUser/l00612380/level3_multi_model/Robot_Play_Chess-master/models//20180129_132912
[INFO] model execute success
Inference time: 1.325ms
[INFO] model execute success
Inference time: 1.136ms
[INFO] model execute success
Inference time: 1.106ms
[INFO] model execute success
Inference time: 1.095ms
[INFO] model execute success
Inference time: 1.096ms
[INFO] model execute success
Inference time: 1.092ms
[INFO] model execute success
Inference time: 1.092ms
[INFO] model execute success
Inference time: 1.105ms
[INFO] model execute success
Inference time: 1.091ms
[INFO] model execute success
Inference time: 1.113ms
[INFO] output data success
Inference average time: 1.125100 ms
Inference average time without first time: 1.102889 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl

```

Batch: 1, shape: 56 * 56* 3，带AIPP，平均推理性能 1.125100 ms。

## 精度测试

在自制测试集上达到99.9%。