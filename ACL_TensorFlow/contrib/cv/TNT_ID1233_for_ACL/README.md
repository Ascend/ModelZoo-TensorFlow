# 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Classification**

**版本（Version）：1.0**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt/pb/om**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

# 概述

TNT（TrackletNet Tracker）是一种性能优秀的跟踪器。

关键技术：

* Tracklet-based Graph Model： 将tracklet作为顶点、将两个tracklets间相似度（的减函数）作为边权的无向图，可以通过顶点聚类算法完成 “tracklet-to-trajectory” 过程；
* Multi-scale TrackletNet：输入两 tracklets，输出其相似度，最大特点是用时域1D滤波器充分利用了 tracklets 的时态信息（temporal info）；
* EG-IOU： 在做帧间detections关联时，使用 Epipolar Geometry（对极几何）对下一帧检测框做最佳预测，从而优化 IOU 算法；

关键能力：

* Graph Model 的设计可以充分使用时域信息、降低计算复杂度等；
* TrackletNet 作为一个统一(unified)的系统，将外观信息(appearance)和时态信息(temporal)合理地结合了起来；注意，传统的时态信息一般是 bbox 的位置、大小、运动等信息，而 TrackletNet 通过时域卷积池化等，挖掘了外观信息中蕴含的时态信息（即外观信息的时域连续性）
* EG 技术可以有效对抗相机运动带来的错误关联问题

参考论文：

[https://arxiv.org/abs/1811.07258](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F1811.07258)

第三方博客地址

[https://blog.csdn.net/qq_42191914/article/details/103619045](https://gitee.com/link?target=https%3A%2F%2Fblog.csdn.net%2Fqq_42191914%2Farticle%2Fdetails%2F103619045)

# 数据集准备

1. 模型训练使用MOT17Det数据集，数据集请用户自行获取。

2. TNT/General/MOT_to_UA_Detrac.m MOT_to_UA_Detrac(gt_file, seq_name, save_folder, img_size)

   **作者提供的典型参数**：MOT_to_UA_Detrac('.txt', '1', 'save_dir', [800,600]); gt_file: MOT gt txt seq_name: save name save_folder: save dir

3. TNT/General/crop_UA_Detrac.m crop_UA_Detrac(gt_path, seq_name, img_folder, img_format, save_folder)

   **作者提供的典型参数**：crop_UA_Detrac('gt_path.mat', '1', 'the folder contains the sequence images', 'jpg', 'save_folder'); gt_path: gtInfo.mat, X,Y,W,H seq_name: name of the sequence img_folder: input image dir img_format: input image format, etc, png, jpg save_folder: cropped image dir

4. TNT/General/create_pair create_pair(dataset_dir, save_dir, num_pair, n_fold)

   **作者提供的典型参**：create_pair('dataset_dir', 'save_dir', 300, 10) dataset_dir: cropped image dir save_dir: output dir num_pair: number of pairs for each fold n_fold: the number of folds 此步骤完成后，需打开所有pairs.txt，在文件头添加一个空行

**注意事项**：

- matlab版本需为2014b及以后的版本，否则会找不到bboxOverlapRatio这个函数（该函数于2014b版本被引进，之前有源码现已无法找到）
- create_pair.m中可能会出现K、pairNum报错的现象，解决方法是将其替换为纯数字就好

**文件安排**： 将所有mat文件，放在一个文件夹下，放在原始数据MOT17Det的下级目录中，（为定位该位置，此处应有两个原始文件夹分别为train和test）而裁剪所得到的图片和pairs.txt仿照lfw数据集格式存放，即存放在类似MOT17Det/train/MOT17-02/img1和MOT17Det/train/MOT17-02/pairs.txt

# 推理过程

* 首先根据模型ckpt文件的存储目录，运行ckpt2pb.py文件，将ckpt文件转为pb文件，冻结模型参数

```
python ckpt2pb.py --ckpt_path ./path/to/ckpt --output_path ./path/to/output_path
```

参数解释：

```
--ckpt_path     ckpt保存目录
--output_path   pb输出文件夹，pb名称默认frozen_model.pb
```


* 数据预处理，运行pic2bin.py文件将.jpg文件转为.bin文件，开始之前请对文件开始部分的参数进行修改：

```python
  python tobin.py 
```

参数含义

```
--MAT_folder              数据预处理步骤生成的mat文件地址
--img_folder              原始数据集地址
--triplet_model           faceNet模型文件，用于对于数据进行预处理
--res_dir                 bin文件存储地址
```

* 在华为云镜像服务器上将pb文件转为om文件

```
atc --model=model/frozen_model.pb --framework=3 --output=res --soc_version=Ascend310 --input_shape="Placeholder:1,1,64,1;Placeholder_1:1,1,64,1;Placeholder_2:1,1,64,1;Placeholder_3:1,1,64,1;Placeholder_4:1,512,64,1;Placeholder_5:1,1,64,2;Placeholder_6:1,512,64,2;Placeholder_8:1" --log=info --out_nodes="add_62:0" --debug_dir=$HOME/module/out/debug_info
```


* 应用msame工具运行模型推理

```
  /root/AscendProjects/tools/msame/out/msame --model /home/HwHiAiUser/zjm/res.om --input /home/HwHiAiUser/zjm/bin/0/batch_X_x.bin,/home/HwHiAiUser/zjm/bin/0/batch_X_y.bin,/home/HwHiAiUser/zjm/bin/0/batch_X_w.bin,/home/HwHiAiUser/zjm/bin/0/batch_X_h.bin,/home/HwHiAiUser/zjm/bin/0/batch_X_a.bin,/home/HwHiAiUser/zjm/bin/0/batch_mask_1.bin,/home/HwHiAiUser/zjm/bin/0/batch_mask_2.bin,/home/HwHiAiUser/zjm/bin/0/keep_prob.bin --output /home/HwHiAiUser/zjm/out/output2/0 --outfmt TXT
```

注意根据上下文，此处的bin数据应有多组，最后根据总体平均acc判定模型效果，此处文件组织为bin/group_num/xxx.bin,请根据自身情况自行组织生成的bin文件及更换参数。

* 得到最终结果，如果想比对准确率，其中与预测结果做对比的是之前生成的batch_y.bin文件，可以运行getAcc.py

```python
  python getAcc.py --predict_path ./predict --bin_path ./bin_path
```

参数解释：

```
--predict_path            msame输出结果总目录，默认为'./predict'
--bin_path                bin文件地址,默认为'./bin'
```

典型目录如下

```
project_path
|-predict
|    |-0
|    |    |-res_output_0.txt
|    |-1
|    |    |-res_output_0.txt
...
|-bin
|    |-0
|    |    |-batch_y.bin
...
```



# 推理模型下载

ckpt模型：
[百度网盘获取链接](https://pan.baidu.com/s/1lzMpTGf910bhWkqUpbAoJw)
提取码：zeu4 

obs地址：obs://cann-id1233/inference/ckpt/

pb模型：
[百度网盘获取链接](https://pan.baidu.com/s/1SXy_iyVQYNAtPZp82yqA0A)
提取码：ub2g 

obs地址：obs://cann-id1233/inference/pb/

om模型：
[百度网盘获取链接](https://pan.baidu.com/s/1eUKpTNeHXcpfRYgAX-liZw)
提取码：2iwo 

obs地址：obs://cann-id1233/inference/om/

# 推理精度

精度结果：

![](.\img\acc.png)

| 迁移模型 | Acc on val | Seconds per Image |
| :------- | ---------- | ----------------- |
| TNT      | 0.95625    | 1.03s             |
