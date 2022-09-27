

# 基本信息

**发布者（Publisher）：Huawei**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.3.29**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的TNT网络训练代码** 

# 概述

TNT（TrackletNet Tracker）是一种性能优秀的跟踪器。

关键技术：

- Tracklet-based Graph Model： 将tracklet作为顶点、将两个tracklets间相似度（的减函数）作为边权的无向图，可以通过顶点聚类算法完成 “tracklet-to-trajectory” 过程；
- Multi-scale TrackletNet：输入两 tracklets，输出其相似度，最大特点是用时域1D滤波器充分利用了 tracklets 的时态信息（temporal info）；
- EG-IOU： 在做帧间detections关联时，使用 Epipolar Geometry（对极几何）对下一帧检测框做最佳预测，从而优化 IOU 算法；

关键能力：

- Graph Model 的设计可以充分使用时域信息、降低计算复杂度等；
- TrackletNet 作为一个统一(unified)的系统，将外观信息(appearance)和时态信息(temporal)合理地结合了起来；注意，传统的时态信息一般是 bbox 的位置、大小、运动等信息，而 TrackletNet 通过时域卷积池化等，挖掘了外观信息中蕴含的时态信息（即外观信息的时域连续性）
- EG 技术可以有效对抗相机运动带来的错误关联问题

参考论文：

[https://arxiv.org/abs/1811.07258](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F1811.07258)

参考实现：

 [https://github.com/GaoangW/TNT](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2FGaoangW%2FTNT)

第三方博客地址

[https://blog.csdn.net/qq_42191914/article/details/103619045](https://gitee.com/link?target=https%3A%2F%2Fblog.csdn.net%2Fqq_42191914%2Farticle%2Fdetails%2F103619045)

适配昇腾 AI 处理器的实现：

[https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/TNT_ID1233_for_TensorFlow

通过Git获取对应commit\_id的代码方法如下：

```
git clone {repository_url}    # 克隆仓库的代码
cd {repository_name}    # 切换到模型的代码仓目录
git checkout  {branch}    # 切换到对应分支
git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

# 训练环境准备

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

   当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

   **表 1** 镜像列表

   <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>

   <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
   </th>
   <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
   </th>
   <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
   </th>
   </tr>
   </thead>
   <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
   </td>
   <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
   </td>
   <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
   </td>
   </tr>
   </tbody>
   </table>

## 快速上手

- 数据集准备

1. 模型训练使用MOT17Det数据集，数据集请用户自行获取。

2. TNT/General/MOT_to_UA_Detrac.m
   MOT_to_UA_Detrac(gt_file, seq_name, save_folder, img_size)
   
   **作者提供的典型参数**：MOT_to_UA_Detrac('.txt', '1', 'save_dir', [800,600]);
   gt_file: MOT gt txt
   seq_name: save name
   save_folder: save dir
   
3. TNT/General/crop_UA_Detrac.m
   crop_UA_Detrac(gt_path, seq_name, img_folder, img_format, save_folder)
   
   **作者提供的典型参数**：crop_UA_Detrac('gt_path.mat', '1', 'the folder contains the sequence images', 'jpg', 'save_folder');
   gt_path: gtInfo.mat, X,Y,W,H
   seq_name: name of the sequence
   img_folder: input image dir
   img_format: input image format, etc, png, jpg
   save_folder: cropped image dir
   
4. TNT/General/create_pair
   create_pair(dataset_dir, save_dir, num_pair, n_fold)
   
   **作者提供的典型参**：create_pair('dataset_dir', 'save_dir', 300, 10)
   dataset_dir: cropped image dir
   save_dir: output dir
   num_pair: number of pairs for each fold
   n_fold: the number of folds
   此步骤完成后，需打开所有pairs.txt，在文件头添加一个空行

**注意事项**：
* matlab版本需为2014b及以后的版本，否则会找不到bboxOverlapRatio这个函数（该函数于2014b版本被引进，之前有源码现已无法找到）

* create_pair.m中可能会出现K、pairNum报错的现象，解决方法是将其替换为纯数字就好

**文件安排**：
	将所有mat文件，放在一个文件夹下，放在原始数据MOT17Det的下级目录中，（为定位该位置，此处应原有两个文件夹分别为train和test）而裁剪所得到的图片和pairs.txt仿照lfw数据集格式存放，即存放在类似MOT17Det/train/MOT17-02/img1和MOT17Det/train/MOT17-02/pairs.txt

## 模型预训练

请用户根据实际路径配置data_dir、lfw_dir、lfw_pairs、pretrained_model等输入路径和log_base_dir、models_base_dir等输出路径参数，利用TNT/src/my_train_tripletloss.py剪裁后的数据和FaceNet网络训练三重态外观模型文件（注意，process_data目录下有多组数据，如MOT17-02,MOT17-04等，故在利用一组数据完成训练后，应及时修改数据地址，再开始下一组数据的训练）

典型参数如下：

```
python 3.7 ${code_dir}/train_tripletloss.py \
          --logs_base_dir ${output_path}/tripletloss_logs/ \
          --models_base_dir ${output_path}/models/MOT17-02/ \
          --data_dir ${data_path}/processed_data/MOT17-02/img/ \
          --lfw_dir ${data_path}/processed_data/MOT17-02/img \
          --lfw_pairs ${data_path}/processed_data/MOT17-02/pairs.txt \
          --pretrained_model ${data_path}/models/pretrained/model-20180402-114759.ckpt-275 \
          --image_size 160 \
          --model_def models.inception_resnet_v1 \
          --optimizer RMSPROP \
          --learning_rate 0.01 \
          --weight_decay 1e-4 \
          --max_nrof_epochs 500 \
          --embedding_size 512 \
          --batch_size 30 \
          --people_per_batch 15 \
          --images_per_person 10 \
          --epoch_size 100
```

(为加快数据传输速度，一般选择在每组数据目录(如dataset/processed_data/MOT17-02/)下，创建model文件存放目录，然后将该组数据地址（如dataset/processed_data/MOT17-02/）设为data path in obs)

示例数据集地址：

URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=kgEgUsFhjaxRtkatN8fRvHyfzFIiPgPMLfABSMULPiLhGFN4hT3ATq1R6KpKf4mVhehzLTPYxD8HJ/GqQ0Z/ATpbVj+cLIc6+J5O98fzKldbzObw4KcTp+z37WFHwqA6hlZ1cBe0NvBwxi4bulpqNoRWxxAzBbmqM4tgC/I/lK7s1u5R0ZcdUYMc8ZB3kN2BJap33XDuZ9qmTJQSOthIoKXCWOIQZi+Aly6fvZoSbYWUcmr5QdXDFOwmXgWIQuEIbOA39Z0qWaJrnvwnwpIBMV39RDma2Kz18hD760ZVLOo4naDtZLXwVQ7Wbgw46saJDDUAoXNkPHOlAA2OlGpoirjpj1P+DLV6GNTa4OJs2itfDfYhY3EMl+NZZC2+1+M0Fer8FK7agCQhZ2rcGgatPlXTEKjd1GTVtOOoIPfPC3A2YzbZBxye+JNwWJO8xbf8E/t7FJcNBp2UWZIefwB0uzSVHTjYxuYF1xghqovSBDwQ76o7ckNTRuI6+9YeNMlaZpj5DWF5pTrM12l43LMSzdeYUlugxLCofFrWH24IpEQ=

提取码:
123456

*有效期至: 2023/03/24 21:20:13 GMT+08:00

# 训练二维跟踪模型

​	在TNT/train_cnn_trajectory_2d.py中所有函数的定义前设置目录路径，（注意triplet_model为之前预训练模型）。根据数据密度更改样本概率（样本概率）sample_prob中的元素数是输入Mat文件的数目。开始时将学习率（lr）设置为1e-3。每2000步，将lr降低10倍，直到达到1e-5。输出模型将存储在save_dir中。运行TNT/my_train_cnn_trajectory_2d.py。（以上步骤已封装到test/train_full_1p.sh中）



# 精度结果

| GPU(199epoch) | NPU(199epoch) |
| ------------- | ------------- |
| 0.790625      | 0.8           |

