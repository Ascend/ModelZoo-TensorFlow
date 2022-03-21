###   **基本信息** 

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Classification**

**版本（Version）：1.0**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt/pb/om**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**


###   **概述** 
CovPoolFer将面部表情分类为不同类别需要捕获面部关键点的区域扭曲。论文提出协方差等二阶统计量能够更好地捕捉区域面部特征中的这种扭曲。论文探索了使用流形网络结构进行协方差合并以改善面部表情识别的好处。特别地，CovPoolFer首次将这种类型的流形网络与传统的卷积网络结合使用，以便以端到端的深度学习方式在各个图像特征映射内进行空间汇集。此外，CovPoolFer利用协方差池来捕捉基于视频的面部表情的每帧特征的时间演变。论文显示了通过在卷积网络层之上堆叠设计的协方差池的流形网络来暂时汇集图像集特征的优点。CovPoolFer在SFEW2.0和RAF数据集上获得了state-of-the-art的结果。

- 相关参考：  
    - 参考论文：[Covariance Pooling for Facial Expression Recognition](https://arxiv.org/abs/1805.04855)
    - 参考实现：[https://github.com/iperov/DeepFaceLab](https://github.com/iperov/DeepFaceLab)


### **数据集准备** 

SFEW数据集百度网盘链接：[百度网盘获取链接](https://pan.baidu.com/s/14zMX4-izJTSL4L6uWySLnw)
提取码：4rdy 

SFEW数据集obs路径：obs://cov/data/SFEW_100/

###  **推理过程**

* 首先根据模型ckpt的存储目录，运行ckpt2pb.py文件，将ckpt文件转为pb文件，冻结模型参数

```python
  python ckpt2pb.py --ckpt_path ./path/to/ckpt --output_path ./path/to/output_path
```
参数解释：
```
--ckpt_path     ckpt保存目录
--output_path   pb输出文件夹，pb名称默认CovPoolFER.pb
```


* 数据预处理，运行pic2bin.py文件将.jpg文件转为.bin文件

```python
  python pic2bin.py --input_image_dir ./path/to/input_image_dir --output_bin_dir ./path/to/output_bin_dir --output_reference_file ./path/to/output_reference_file
```

参数解释：
```
--input_image_dir         数据（图片）保存目录
--output_bin_dir          bin输出文件夹
--output_reference_file   bin对应标签输出文件
```

* 在华为云镜像服务器上将pb文件转为om文件

```
 atc --model=./path/to/pb --framework=3 --output=./path/to/om_dir --soc_version=Ascend310 --input_shape="inputImage:128,100,100,3" --log=info --out_nodes="output:0"
```


* 应用msame工具运行模型推理

```
  ./msame --model ./path/to/om --input ./path/to/input/bin --output ./path/to/output/txt --outfmt TXT --loop 1
```

* 得到最终结果，如果想比对准确率，可以运行getAcc.py

```python
  python getAcc.py --input_om_out ./path/to/om_out --input_reference_out ./path/to/output_inference_file
```

参数解释：
```
--input_om_out            msame输出结果目录
--input_reference_out     bin对应标签输出文件
```

###  **推理模型下载**

ckpt模型：
[百度网盘获取链接](https://pan.baidu.com/s/1mVKxTINOgvG6syoLXJwjUg )
提取码：cvua 

obs地址：obs://cov/models/ckpt/

pb模型：
[百度网盘获取链接](https://pan.baidu.com/s/1Jf8GauHoPjnJi3m2ZYGqfA )
提取码：am89 

obs地址：obs://cov/models/pb/

om模型：
[百度网盘获取链接](https://pan.baidu.com/s/19ulcJhjL7awGlzGok683eQ )
提取码：xejs 

obs地址：obs://cov/models/om/

###  **推理性能及精度**

![Image](pics/performance.png)

| 迁移模型   | Acc on val | Seconds per Image |
| :--------- | ---------- | ----------------- |
| CovPoolFER | 0.316      | 31.09ms           |