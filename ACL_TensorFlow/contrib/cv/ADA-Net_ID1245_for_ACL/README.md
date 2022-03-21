# ADA-Net

## 模型功能

ADA-Net采用了一种对抗性训练策略来最小化标记数据和非标记数据之间的分布距离。用来识别图片中的数字。

## 原始模型

参考实现 ：https://github.com/qinenergy/adanet

由自己训练的ckpt生成pb模型：

```bash
python pbmake.py --model_path= "the path of ckpt"
```

pb模型获取链接：

华为obs链接：https://adanet1.obs.cn-north-4.myhuaweicloud.com:443/model/svhn.pb?AccessKeyId=CHYIIBYL450CCJXFJNBB&Expires=1668846057&Signature=nDs56HT3dSYs70qndkwKhqD0Hiw%3D

百度网盘链接：https://pan.baidu.com/s/1Hl5gQzmjhEwC0_M529NlfQ 
提取码：x3sy

## om模型

om模型
- 华为obs下载地址：https://adanet1.obs.cn-north-4.myhuaweicloud.com:443/model/svhn_v1.om?AccessKeyId=CHYIIBYL450CCJXFJNBB&Expires=1668846252&Signature=0Syzw8p%2BwEG4cygeh1xWzXAyyko%3D
- 百度网盘链接：https://pan.baidu.com/s/1a2STTEssI8MfnVqIb5k3jA 
提取码：y0f6

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  
- [ATC工具使用指导 - Atlas 200dk](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 
- [ATC工具使用环境搭建_昇腾CANN社区版(5.0.3.alpha002)(推理)_ATC模型转换_华为云](https://support.huaweicloud.com/atctool-cann503alpha2infer/atlasatc_16_0004.html)

命令行示例：

```ba
atc --model=$HOME/adanet/svhn.pb --framework=3 --output=./svhn_v1  --soc_version=Ascend310   --input_shape="input:1,32,32,3"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
~/msame/msame --model ~/adanet/svhn_v1.om --output ~/msame/out-10-31  --outfmt TXT --loop 1
```


Batch: 1, shape: 32,32,3， 推理性能 1.57ms

## 精度测试
- 生成数据
```bash
python svhn_bin.py --data_dir = 'the path of svhn data'
```
生成的bin文件存储在当前目录的image文件夹下。

- om模型推理

```bash
~/msame/msame --model ~/infer/svhn_v1.om  --input "/home/HwHiAiUser/adanet/image/" --output ~/adanet/output/  --outfmt TXT
```
## 推理精度

|      | 推理  | 论文  |
| ---- | ----- | ----- |
| ACC  | 94.40 | 95.38 |



