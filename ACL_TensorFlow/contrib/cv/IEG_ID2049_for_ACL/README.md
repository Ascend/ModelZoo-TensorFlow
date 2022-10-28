<h2 id="概述.md">概述</h2>
本文提出了一个高效率的噪声标签训练方法。


- 参考论文：
[Distilling Effective Supervision from Severe Label Noise](https://arxiv.org/pdf/1910.00701.pdf),
    CVPR2020
- 参考实现：
https://github.com/google-research/google-research/tree/master/ieg

- 适配昇腾 AI 处理器的实现：
https://gitee.com/lwrstudy/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Slot-Attention_ID2028_for_TensorFlow


<h2 id="概述.md">原始模型</h2>

obs地址：obs://lwr-npu/tl/checkp


步骤一:将checkpoint.ckpt-85329转化成IEG.pb
通过代码freeze_graph将ckpt转成pb



<h2 id="概述.md">pb模型</h2>

```
IEG.pb
```
obs地址：obs://lwr-npu/tl/IEG.pb


<h2 id="概述.md">om模型</h2>

转IEG.pb到hpiegmodel.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=./IEG.pb --input_shape=“input:100,3,32,32” --framework=3 --output=hpiegmodel --soc_version=Ascend910A --precision_mode=force_fp32 --op_select_implmode=high_precision
```

成功转化成hpiegmodel.om

hpiegmodel.om的obs地址：obs://lwr-npu/tl/hpiegmodel.om


<h2 id="概述.md">数据集转换bin</h2>
1.使用convertimg.py将数据集转为png，代码中的src为数据集源路径，dest为png数据集输出路径
2.参考 https://gitee.com/ascend/tools/tree/master/img2bin, 获取msame推理工具及使用方法。
这里对该代码库中的img2bin进行了少量修改，加入了数据预处理
使用img2bin.py将png数据集转为bin格式，img2bin.py为tools-master中的代码文件,
```
python3 img2bin.py -i /mnt/home/test_user01/toimg -w 32 -h 32 -f RGB -a NHWC -t float32  -c [1,1,1] -o ./imgout 
```
3.使用merge.py将bin格式的数据集合并为一个batch
转换后的bin文件见obs地址:obs://lwr-npu/tl/newinputbin




<h2 id="概述.md">使用msame工具推理</h2>

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

使用msame推理工具，参考如下命令，发起推理测试：

```
./msame --model "/mnt/home/test_user01/hpiegmodel.om" --input "/home/test_user01/slottl/tools-master/img2bin/newinput.bin" --output "./iegout/" --outfmt TXT --loop 1
```









