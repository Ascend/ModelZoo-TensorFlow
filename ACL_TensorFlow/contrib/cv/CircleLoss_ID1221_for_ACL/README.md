# CircleLoss

**Circle Loss: A Unified Perspective of Pair Similarity Optimization**

Yifan Sun et al.

CVPR 2020

ArXiv: [https://arxiv.org/abs/2002.10857](https://arxiv.org/abs/2002.10857)

本文提出了一种新的pair-wise的相似度优化损失函数，能够在学习过程中自适应地调整对不同类型相似度的惩罚程度，从而达到更加高效学习效果。

![image-20210906114857004](https://picbed-1301760901.cos.ap-guangzhou.myqcloud.com/image-20210906114857004.png)


- 适配昇腾 AI 处理器的实现：
    
        
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/CircleLoss_ID1221_for_TensorFlow
        




- 精度

| -   | 论文    | GPU （v100） | NPU   |
|-----|-------|------------|-------|
| acc | 0.997 | 0.948      | 0.951 |

- 性能

| batchsize  |image_size| GPU （v100）   | NPU          |
|---|---|--------------|--------------|
| 30  | 112 x 112| 0.076 s/step | 0.046 s/step |

   
## 1. 依赖库安装
见requirements.txt， 需要安装 numpy， scikit-image， tensorflow等库。

## 2.获取测试集

    obs链接:obs://cann-id1221/dataset/lfw-deepfunneled_align.zip
    下载数据集后解压到工作目录

## 3. 转换测试集为bin
- 修改img2bin.py文件中的lfw_path为lfw数据集路径 

    ```
      lfw_path="./lfw-deepfunneled_align
    ```
- 设置bin文件输出路径

    ```
      output_path="./lfw_bin
    ```
- 执行程序

    ```
      python img2bin.py
    ```
## 4. 获取pb模型文件
      obs链接: obs://cann-id1221/ACL/Model.pb
## 5.pb转om
      atc --model=/root/Model.pb --framework=3 --output=/root/Model --soc_version=Ascend310 --input_shape="input:1,112,112,3" --log=info --out_nodes="Reshape:0"
## 6.msame推理
      ./msame --model "/root/Model.om" --input "/root/lfw_bin" --output "/root/output" --outfmt TXT
## 7.计算推理准确率
- 配置cal_acc.py文件中output_path参数

      output_path=./output/2022713_17_55_11_543761
- 计算准确率

      python cal_acc.py

