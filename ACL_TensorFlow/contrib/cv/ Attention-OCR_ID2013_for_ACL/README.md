**描述（Description）：基于TensorFlow框架的Attention-OCR自然场景文本检测识别网络训练代码** 

<h2 id="概述.md">概述</h2>

Attention-OCR是一个基于卷积神经网络CNN、循环神经网络RNN以及一种新颖的注意机制的自然场景文本检测识别网络。

- 参考论文：

    ["Attention-based Extraction of Structured Information from Street View
    Imagery"](https://arxiv.org/abs/1704.03549)


- 参考实现：[models/research/attention_ocr at master · tensorflow/models · GitHub](https://github.com/tensorflow/models/tree/master/research/attention_ocr)


- 适配昇腾 AI 处理器的实现：

  [TensorFlow/contrib/cv/Attention-OCR_ID2013_for_TensorFlow · Ypo6opoc/ModelZoo-TensorFlow - 码云 - 开源中国 (gitee.com)](https://gitee.com/ypo6opoc/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Attention-OCR_ID2013_for_TensorFlow)



<h2 id="概述.md">原始模型</h2>

obs地址：obs://cann-2021-10-21/inference/ckpt_file/



步骤一:将model.ckpt-1000000转化成attention_ocr.pb
通过代码freeze_pb将ckpt转成pb



<h2 id="概述.md">pb模型</h2>

```
attention_ocr.pb
```
obs地址：obs://cann-2021-10-21/inference/pb_file/



<h2 id="概述.md">om模型</h2>

转attention_ocr.pb到attention_ocr.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
./atc --input_shape="input:1,150,600,3" --model="/home/Attention-OCR/pb_file/attention_ocr.pb" --framework=3 --soc_version=Ascend310 --input_format=NHWC --output="/home/Attention-OCR/modelzoo/attention_ocr/Ascend310/attention_ocr" --check_report=/home/Attention-OCR/modelzoo/attention_ocr/Ascend310/network_analysis.report
```

成功转化成attention_ocr.om

attention_ocr.om的obs地址：obs://cann-2021-10-21/inference/om_file/



<h2 id="概述.md">数据集转换bin</h2>

使用png2bin.py将png格式的测试图片转为bin格式。

转换后的bin文件见obs地址:obs://cann-2021-10-21/inference/bin_file/



<h2 id="概述.md">使用msame工具推理</h2>

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

使用msame推理工具，参考如下命令，发起推理测试：

```
./msame --model /home/Attention-OCR/om_file/attention_ocr.om --input /home/Attention-OCR/bin_file/fsns_train_00.bin --output /home/Attention-OCR/out/output1 --outfmt TXT --loop 2
```









