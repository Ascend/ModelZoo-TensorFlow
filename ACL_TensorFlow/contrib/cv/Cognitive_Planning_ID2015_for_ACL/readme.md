#  LEARNING-TO-SEE-IN-THE-DARK

## 离线推理

### 1. 原始模型转pb

```
python3.7 ./ckpt2pb.py  
```

转换好的pb模型，obs地址：
obs://sid-obs/inference/sid.pb

### 2. pb转om模型

使用atc模型转换工具转换pb模型到om模型

```
atc --model=/home/HwHiAiUser/AscendProjects/SID/pb_model/sid.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/SID/ --soc_version=Ascend310 --input_shape="Placeholdr:1,1424,2128,4; " --log=info --out_nodes="DepthToSpace:0"  
```

转换好的OM模型，obs地址：
obs://sid-obs/inference/sid.om

### 3. 数据处理

对输入的Bayer图片按照R，G，B分为4个通道，得到(1424,2128,4)的输入图片和(2848,4256,3)的真值图片。同时，将处理过的图片从ARW格式变为BIN格式。

```
python3.7 ./data_pre.py
```

生成的输入数据bin文件，obs地址：
obs://sid-obs/inference/Sony/long
obs://sid-obs/inference/Sony/short

### 4. 准备msame推理

参考[msame](https://gitee.com/ascend/modelzoo/wikis/离线推理案例/离线推理工具msame使用案例)

### 5. om模型推理

使用如下命令进行性能测试：

```
./msame --model /home/HwHiAiUser/AscendProjects/SID/sid.om --input /home/HwHiAiUser/AscendProjects/SID/short --output /home/HwHiAiUser/AscendProjects/SID/out/ --outfmt BIN  
```

测试结果如下：

```
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
Inference average time : 426.47 ms
Inference average time without first time: 426.47 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

单张图片的平均推理时间为426.47ms

### 6. om精度测试

使用如下命令进行精度测试：

```
python3.7.5 ./eval_om.py
```

测试结果如下：

|      | 原论文 | GPU   | NPU   | 离线推理 |
| ---- | ------ | ----- | ----- | -------- |
| PSNR | 28.88  | 27.63 | 28.26 | 28.64    |
| SSIM | 0.78   | 0.719 | 0.715 | 0.773    |

离线推理精度达标
om推理输出bin文件，obs地址：
obs://sid-obs/inference/Sony/om_out