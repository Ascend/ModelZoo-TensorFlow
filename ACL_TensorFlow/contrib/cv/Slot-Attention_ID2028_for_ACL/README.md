<h2 id="概述.md">概述</h2>
提出了Slot Attention 模块，建立了感知表征 (perceptual representations, 如CNN 输出) 与 slots 之间的桥梁 (Feature map/Grid → Set of slots)
	

- 参考论文：

    @article{locatello2020object,
    title={Object-Centric Learning with Slot Attention},
    author={Locatello, Francesco and Weissenborn, Dirk and Unterthiner, Thomas and Mahendran, Aravindh and Heigold, Georg and Uszkoreit, Jakob and Dosovitskiy, Alexey and Kipf, Thomas},
    journal={arXiv preprint arXiv:2006.15055},
    year={2020}
}

- 参考实现：

   https://github.com/google-research/google-research/tree/master/slot_attention

- 适配昇腾 AI 处理器的实现：
    
https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Slot-Attention_ID2028_for_TensorFlow

<h2 id="概述.md">原始模型</h2>

obs地址：obs://lwr-slot-npu/slottl/newslotmodel.pb


步骤一:将转化成IEG.pb
通过代码keras_frozen_graph将ckpt-499000转成pb
ckpt的obs地址：obs://lwr-slot-npu/slottl/
该目录中以checkpoint开头的四个文件


<h2 id="概述.md">pb模型</h2>

```
newslotmodel.pb
```
obs地址：obs://lwr-slot-npu/slottl/newslotmodel.pb


<h2 id="概述.md">om模型</h2>

转newslotmodel.pb到slotmodel.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=./newslotmodel.pb --input_shape="input:64, 128, 128, 3" --framework=3 --output=slotmodel --soc_version=Ascend910A --precision_mode=force_fp32 --op_select_implmode=high_precision
```

成功转化成slotmodel.om

slotmodel.om的obs地址:obs://lwr-slot-npu/slottl/slotmodel.om



<h2 id="概述.md">使用msame工具推理</h2>

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

使用msame推理工具，参考如下命令，发起推理测试：

```
./msame --model "slotmodel.om"  --output "./" --outfmt TXT --loop 1
```