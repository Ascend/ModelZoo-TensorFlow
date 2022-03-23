# agegenderLMTCNN

## 模型功能

agegenderLMTCNN是一种同时预测图片中人的年龄和性别的网络。

## 原始模型

参考实现 ：https://github.com/ivclab/agegenderLMTCNN

由自己训练的ckpt生成pb模型：

```bash
python pbmake.py --model_path= "the path of ckpt"
```

ckpt模型获取链接：链接：https://pan.baidu.com/s/1SgU8T8hD3pTny9C2RxAslw 
提取码：wadz

pb模型获取链接：链接：https://pan.baidu.com/s/19kzo-XKGixbDnxmS_5tmUw 
提取码：rn0r

## om模型

om模型盘链接：链接：https://pan.baidu.com/s/1n-tAvw6KgnKt_9YkBQJwpg 
提取码：4rrm

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  

- [ATC工具使用指导 - Atlas 200dk](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 
- [ATC工具使用环境搭建_昇腾CANN社区版(5.0.3.alpha002)(推理)_ATC模型转换_华为云](https://support.huaweicloud.com/atctool-cann503alpha2infer/atlasatc_16_0004.html)

命令行示例：

```bash
atc --model=$HOME/lmtcnn/lmtcnn_npu.pb --framework=3 --output=./lmtcnn_npu  --soc_version=Ascend310   --input_shape="input:1,227,227,3"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```bash
~/msame/msame --model ~/lmtcnn/lmtcnn_npu.om --output ~/msame/output  --outfmt TXT --loop 1
```


Batch: 1, shape: 227,227,3， 推理性能 1.57ms

## 精度测试

- 生成数据

参考链接LMTCNN训练项目：https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/LMTCNN_ID1278_for_TensorFlow

运行multipreproc.py文件生成对应的tfrecord文件

```bash
python svhn_bin.py --data_dir = 'the path of tfrecord'
```

生成的bin文件存储在当前目录的imagebin文件夹下。对应生成的标签npy文件存储在当前目录的label文件夹下。

- om模型推理

```bash
~/msame/msame --model ~/infer/lmtcnn_npu.om  --input "/home/HwHiAiUser/lmtcnn/imagebin/"  --output ~/lmtcnn/output/  --outfmt TXT
```

- 推理结果分析
```bash
python result.py --age_label_dir = 'the path of age label'  --gender_label_dir = 'the path of gender label'   --output_dir = 'the folder of the output txt file' 
```

## 推理精度

| ACC      | 推理  | GPU   | NPU   |
| -------- | ----- | ----- | ----- |
| ageTop_1 | 39.27 | 37.08 | 36.43 |
| ageTop_2 | 77.04 | 60.61 | 58.50 |
| gender   | 80.63 | 80.89 | 78.52 |

