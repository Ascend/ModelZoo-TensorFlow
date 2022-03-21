

<h2 id="概述.md">概述</h2>

Multi-level Spatially-Pooled (MLSP) features extracted from ImageNet pre-trained Inception-type networks are used to train aesthetics score (MOS) predictors on the Aesthetic Visual Analysis (AVA) database. The code shows how to train models based on both narrow and wide MLSP features.

- This is part of the code for the paper "Effective Aesthetics Prediction with Multi-level Spatially Pooled Features". Please cite the following paper if you use the code:
    
    ```
    @inproceedings{hosu2019effective,
  title={Effective Aesthetics Prediction with Multi-level Spatially Pooled Features},
  author={Hosu, Vlad and Goldlucke, Bastian and Saupe, Dietmar},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9375--9383},
  year={2019}}
    ```

<h2 id="快速上手.md">快速上手</h2>
<h3>方案一</h3>（从外网下载数据集之后，配置相应的路径）
- 数据集准备
1. AVA DATASSET -- A Large-Scale Database for Aesthetic Visual Analysis

2. 下载链接：https://github.com/mtobeiyf/ava_downloader

3. 数据集下载后，放入对应的目录下，在训练脚本中指定数据集路径，可正常使用。


<h3>方案二</h3>（下载OBS桶上面存储的数据集，即不使用默认解析好的特征）

运行extract_mlsp.py,运行之后会进行特征解析，得到一个i1[orig]_lfinal_o1[5,5,16928]_r1.h5'的特征，也就是训练时所需的输入文件。
接着把该运行train_mlsp_wide.py，生成最终的模型。



<h3>方案三</h3>（跳过方案二的特称解析部分，使用默认解析好的特征，但是数据集也是需要的布置的）
下载我预先训练好的i1[orig]_lfinal_o1[5,5,16928]_r1.h5的特征，之后配置路径之后，直接运行train_mlsp_wide.py文件即可，也会得到最终的模型。

## 文件夹及文件夹下的文件解释和说明<a name="section08421615141513"></a>

整个项目应该有同一目录下的五个文件夹组成
![输入图片说明](../../../../../image.png)


dataset-mnist:该文件夹下存放的是数据集

features：该文件夹下面的irnv2_mlsp_wide_orig文件夹中存放extract_mlsp.py对图片特征解析得到的特征文件i1[orig]_lfinal_o1[5,5,16928]_r1.h5

metadata：该文件夹下存放的是AVA_data_official_test.csv，该文件保存数据集的名称及其他的一些label

models：该文件夹下面的irnv2_mlsp_wide_orig文件夹里面存放着最终训练出来的模型

pretrain_model：存放着keras的预训练的网络模型，由于模型较大、网络因素等原因，十分容易出现下载失败的情况，导致代码运行失败，这里建议把这两个模型放到/home/ma-user/.keras/models下，这样就可以自动加载该模型，而无需下载。
## 脚本和示例代码<a name="section08421615141513"></a>

```
├── extract_mlsp.py                         //从AVA图像中解析 MLSP 特征并保存到 HDF5 文件；允许存储每个图像增强。
├── train_mlsp_narrow.py, train_mlsp_narrow_aug.py//在存储特征（裁剪、翻转）之前应用或不应用增强，对来自 InceptionResNet-v2 的窄 MLSP 特征（1×1×16k）进行训练。
├── train_mlsp_wide.py                    //训练来自 InceptionResNet-v2 的宽 MLSP 特征 (5×5×16k)，无需增强
├── predict_mlsp_wide.py                    //组装直接从图像（而不是保存的 MLSP 特征）预测分数的预训练模型。           
├── README.md                                  //代码说明文档
├── kutils
│    ├──generic.py                             //H5Helper：管理 HDF5 文件中的命名数据集，在 Keras 生成器中为我们服务。
|                                              //pretty: 漂亮打印字典类型对象。
|                                              //ShortNameBuilder：用于构建包含多个参数的短（文件）名称的实用程序。

│    ├──image_utils.py                         //ImageAugmenter：为训练 Keras 模型创建自定义图像增强函数。
                                               //read_image, read_image_batch: 用于操作图像的实用函数。
                                               
│    ├──model_helper.py                        //ModelHelper：简化 Keras 对回归模型的默认使用的包装类。

│    ├──applications.py                        //model_inception_multigap, model_inceptionresnet_multigap: 用于提取 MLSP 窄特征的模型定义
                                               //model_inception_pooled, model_inceptionresnet_pooled: 用于提取 MLSP 宽特征的模型定义  
                                                 
│    ├──generators.py                          //DataGeneratorDisk, DataGeneratorHDF5: 用于磁盘图像和 HDF5 存储特征/图像的 Keras 生成器
```

## 脚本参数<a name="section6669162441511"></a>




```
--Batch size: 128
--Learning rate(LR): 0.0001，每20个epoch除以10
--Optimizer: Adam
--Train epoch: 20*3
```

## 训练过程<a name="section1589455252218"></a>

1. Pycharm布置好modelarts环境之后，点击Apply and Run执行代码。
2. 需要执行的代码：train_mlsp_wide.py
3. 训练脚本log中包括如下信息。

```
1605/1748 [==========================>...] - ETA: 1:53 - loss: 0.3157 - mean_absolute_error: 0.4418 - plcc_tf: 0.6928
1606/1748 [==========================>...] - ETA: 1:52 - loss: 0.3157 - mean_absolute_error: 0.4418 - plcc_tf: 0.6928
1607/1748 [==========================>...] - ETA: 1:51 - loss: 0.3156 - mean_absolute_error: 0.4418 - plcc_tf: 0.6928
1608/1748 [==========================>...] - ETA: 1:50 - loss: 0.3157 - mean_absolute_error: 0.4418 - plcc_tf: 0.6929
1609/1748 [==========================>...] - ETA: 1:49 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6929
1610/1748 [==========================>...] - ETA: 1:49 - loss: 0.3156 - mean_absolute_error: 0.4417 - plcc_tf: 0.6929
1611/1748 [==========================>...] - ETA: 1:48 - loss: 0.3156 - mean_absolute_error: 0.4417 - plcc_tf: 0.6929
1612/1748 [==========================>...] - ETA: 1:47 - loss: 0.3156 - mean_absolute_error: 0.4417 - plcc_tf: 0.6929
1613/1748 [==========================>...] - ETA: 1:46 - loss: 0.3156 - mean_absolute_error: 0.4417 - plcc_tf: 0.6930
1614/1748 [==========================>...] - ETA: 1:45 - loss: 0.3156 - mean_absolute_error: 0.4417 - plcc_tf: 0.6929
1615/1748 [==========================>...] - ETA: 1:45 - loss: 0.3156 - mean_absolute_error: 0.4417 - plcc_tf: 0.6929
1616/1748 [==========================>...] - ETA: 1:44 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6929
1617/1748 [==========================>...] - ETA: 1:43 - loss: 0.3157 - mean_absolute_error: 0.4418 - plcc_tf: 0.6928
1618/1748 [==========================>...] - ETA: 1:42 - loss: 0.3157 - mean_absolute_error: 0.4418 - plcc_tf: 0.6928
1619/1748 [==========================>...] - ETA: 1:41 - loss: 0.3157 - mean_absolute_error: 0.4418 - plcc_tf: 0.6928
1620/1748 [==========================>...] - ETA: 1:41 - loss: 0.3157 - mean_absolute_error: 0.4418 - plcc_tf: 0.6928
1621/1748 [==========================>...] - ETA: 1:40 - loss: 0.3157 - mean_absolute_error: 0.4418 - plcc_tf: 0.6928
1622/1748 [==========================>...] - ETA: 1:39 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1623/1748 [==========================>...] - ETA: 1:38 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1624/1748 [==========================>...] - ETA: 1:38 - loss: 0.3156 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1625/1748 [==========================>...] - ETA: 1:37 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1626/1748 [==========================>...] - ETA: 1:36 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1627/1748 [==========================>...] - ETA: 1:35 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1628/1748 [==========================>...] - ETA: 1:34 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1629/1748 [==========================>...] - ETA: 1:34 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1630/1748 [==========================>...] - ETA: 1:33 - loss: 0.3156 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1631/1748 [==========================>...] - ETA: 1:32 - loss: 0.3156 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1632/1748 [===========================>..] - ETA: 1:31 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6927
1633/1748 [===========================>..] - ETA: 1:30 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6927
1634/1748 [===========================>..] - ETA: 1:30 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1635/1748 [===========================>..] - ETA: 1:29 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1636/1748 [===========================>..] - ETA: 1:28 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1637/1748 [===========================>..] - ETA: 1:27 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6927
1638/1748 [===========================>..] - ETA: 1:26 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6927
1639/1748 [===========================>..] - ETA: 1:26 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6927
1640/1748 [===========================>..] - ETA: 1:25 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6927
1641/1748 [===========================>..] - ETA: 1:24 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6927
1642/1748 [===========================>..] - ETA: 1:23 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6927
1643/1748 [===========================>..] - ETA: 1:22 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6927
1644/1748 [===========================>..] - ETA: 1:22 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6927
1645/1748 [===========================>..] - ETA: 1:21 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6927
1646/1748 [===========================>..] - ETA: 1:20 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6927
1647/1748 [===========================>..] - ETA: 1:19 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1648/1748 [===========================>..] - ETA: 1:19 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1649/1748 [===========================>..] - ETA: 1:18 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1650/1748 [===========================>..] - ETA: 1:17 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6928
1651/1748 [===========================>..] - ETA: 1:16 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6929
1652/1748 [===========================>..] - ETA: 1:15 - loss: 0.3157 - mean_absolute_error: 0.4417 - plcc_tf: 0.6929
1653/1748 [===========================>..] - ETA: 1:15 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6929
1654/1748 [===========================>..] - ETA: 1:14 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6929
1655/1748 [===========================>..] - ETA: 1:13 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6929
1656/1748 [===========================>..] - ETA: 1:12 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6929
1657/1748 [===========================>..] - ETA: 1:11 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6928
1658/1748 [===========================>..] - ETA: 1:11 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6928
1659/1748 [===========================>..] - ETA: 1:10 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6928
1660/1748 [===========================>..] - ETA: 1:09 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6928
1661/1748 [===========================>..] - ETA: 1:08 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6928
1662/1748 [===========================>..] - ETA: 1:07 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6928
1663/1748 [===========================>..] - ETA: 1:07 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6928
1664/1748 [===========================>..] - ETA: 1:06 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6928
1665/1748 [===========================>..] - ETA: 1:05 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6928
1666/1748 [===========================>..] - ETA: 1:04 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6928
1667/1748 [===========================>..] - ETA: 1:03 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6928
1668/1748 [===========================>..] - ETA: 1:03 - loss: 0.3156 - mean_absolute_error: 0.4416 - plcc_tf: 0.6928
1669/1748 [===========================>..] - ETA: 1:02 - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1670/1748 [===========================>..] - ETA: 1:01 - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1671/1748 [===========================>..] - ETA: 1:00 - loss: 0.3156 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1672/1748 [===========================>..] - ETA: 1:00 - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1673/1748 [===========================>..] - ETA: 59s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929 
1674/1748 [===========================>..] - ETA: 58s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1675/1748 [===========================>..] - ETA: 57s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1676/1748 [===========================>..] - ETA: 56s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1677/1748 [===========================>..] - ETA: 56s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1678/1748 [===========================>..] - ETA: 55s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1679/1748 [===========================>..] - ETA: 54s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1680/1748 [===========================>..] - ETA: 53s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1681/1748 [===========================>..] - ETA: 52s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1682/1748 [===========================>..] - ETA: 52s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1683/1748 [===========================>..] - ETA: 51s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1684/1748 [===========================>..] - ETA: 50s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1685/1748 [===========================>..] - ETA: 49s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1686/1748 [===========================>..] - ETA: 48s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1687/1748 [===========================>..] - ETA: 48s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1688/1748 [===========================>..] - ETA: 47s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1689/1748 [===========================>..] - ETA: 46s - loss: 0.3156 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1690/1748 [============================>.] - ETA: 45s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1691/1748 [============================>.] - ETA: 45s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1692/1748 [============================>.] - ETA: 44s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1693/1748 [============================>.] - ETA: 43s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1694/1748 [============================>.] - ETA: 42s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1695/1748 [============================>.] - ETA: 41s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1696/1748 [============================>.] - ETA: 41s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1697/1748 [============================>.] - ETA: 40s - loss: 0.3156 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1698/1748 [============================>.] - ETA: 39s - loss: 0.3156 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1699/1748 [============================>.] - ETA: 38s - loss: 0.3156 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1700/1748 [============================>.] - ETA: 37s - loss: 0.3156 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1701/1748 [============================>.] - ETA: 37s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1702/1748 [============================>.] - ETA: 36s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1703/1748 [============================>.] - ETA: 35s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1704/1748 [============================>.] - ETA: 34s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1705/1748 [============================>.] - ETA: 33s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1706/1748 [============================>.] - ETA: 33s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1707/1748 [============================>.] - ETA: 32s - loss: 0.3155 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1708/1748 [============================>.] - ETA: 31s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1709/1748 [============================>.] - ETA: 30s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1710/1748 [============================>.] - ETA: 30s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1711/1748 [============================>.] - ETA: 29s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1712/1748 [============================>.] - ETA: 28s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1713/1748 [============================>.] - ETA: 27s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1714/1748 [============================>.] - ETA: 26s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1715/1748 [============================>.] - ETA: 26s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1716/1748 [============================>.] - ETA: 25s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6931
1717/1748 [============================>.] - ETA: 24s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1718/1748 [============================>.] - ETA: 23s - loss: 0.3154 - mean_absolute_error: 0.4415 - plcc_tf: 0.6931
1719/1748 [============================>.] - ETA: 22s - loss: 0.3154 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1720/1748 [============================>.] - ETA: 22s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1721/1748 [============================>.] - ETA: 21s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1722/1748 [============================>.] - ETA: 20s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1723/1748 [============================>.] - ETA: 19s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1724/1748 [============================>.] - ETA: 18s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1725/1748 [============================>.] - ETA: 18s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1726/1748 [============================>.] - ETA: 17s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1727/1748 [============================>.] - ETA: 16s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1728/1748 [============================>.] - ETA: 15s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1729/1748 [============================>.] - ETA: 15s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1730/1748 [============================>.] - ETA: 14s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1731/1748 [============================>.] - ETA: 13s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1732/1748 [============================>.] - ETA: 12s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1733/1748 [============================>.] - ETA: 11s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1734/1748 [============================>.] - ETA: 11s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1735/1748 [============================>.] - ETA: 10s - loss: 0.3154 - mean_absolute_error: 0.4414 - plcc_tf: 0.6930
1736/1748 [============================>.] - ETA: 9s - loss: 0.3154 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930 
1737/1748 [============================>.] - ETA: 8s - loss: 0.3154 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1738/1748 [============================>.] - ETA: 7s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1739/1748 [============================>.] - ETA: 7s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1740/1748 [============================>.] - ETA: 6s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1741/1748 [============================>.] - ETA: 5s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6930
1742/1748 [============================>.] - ETA: 4s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1743/1748 [============================>.] - ETA: 3s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1744/1748 [============================>.] - ETA: 3s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1745/1748 [============================>.] - ETA: 2s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1746/1748 [============================>.] - ETA: 1s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1747/1748 [============================>.] - ETA: 0s - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929
1748/1748 [==============================] - 1401s 801ms/step - loss: 0.3155 - mean_absolute_error: 0.4415 - plcc_tf: 0.6929 - val_loss: 0.2507 - val_mean_absolute_error: 0.3874 - val_plcc_tf: 0.7249
打印最终的模型存储路径：/cache/models/data_url_0/bn2_bsz128_do[0.25,0.25,0.5,0]_ds[AVA_data_official_test.csv]_fc1[2048]_i1[5,5,16928]_im[orig]_lMSE_mon[val_plcc_tf]_o1[1]_best_weights.h5
/cache/models/data_url_0/bn2_bsz128_do[0.25,0.25,0.5,0]_ds[AVA_data_official_test.csv]_fc1[2048]_i1[5,5,16928]_im[orig]_lMSE_mon[val_plcc_tf]_o1[1]_best_weights.h5
Model weights loaded: data_url_0/bn2_bsz128_do[0.25,0.25,0.5,0]_ds[AVA_data_official_test.csv]_fc1[2048]_i1[5,5,16928]_im[orig]_lMSE_mon[val_plcc_tf]_o1[1]_best_weights.h5
Testing model
Model outputs: ['head_out']
2021-10-12 14:12:01.227378: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SOURCE is null.
2021-10-12 14:12:01.227444: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.
2021-10-12 14:12:01.228316: W tf_adapter/util/infershape_util.cc:337] The shape of node branch_1x1_bn/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:12:01.228401: W tf_adapter/util/infershape_util.cc:337] The shape of node branch_pool_bn/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:12:01.228446: W tf_adapter/util/infershape_util.cc:337] The shape of node 3x3_1x1_bn/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:12:01.228553: W tf_adapter/util/infershape_util.cc:337] The shape of node branch_1x1_bn/cond/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:12:01.228586: W tf_adapter/util/infershape_util.cc:337] The shape of node 3x3_1x1_bn/cond/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:12:01.228616: W tf_adapter/util/infershape_util.cc:337] The shape of node branch_pool_bn/cond/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:12:01.228761: W tf_adapter/util/infershape_util.cc:337] The shape of node 3x3_1x3_bn/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:12:01.228804: W tf_adapter/util/infershape_util.cc:337] The shape of node 3x3_3x1_bn/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:12:01.228835: W tf_adapter/util/infershape_util.cc:337] The shape of node 3x3_1x3_bn/cond/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:12:01.228873: W tf_adapter/util/infershape_util.cc:337] The shape of node 3x3_3x1_bn/cond/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:12:01.229441: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node group_deps_3 is null.
2021-10-12 14:14:00.545852: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SOURCE is null.
2021-10-12 14:14:00.545930: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.
2021-10-12 14:14:00.546843: W tf_adapter/util/infershape_util.cc:337] The shape of node branch_1x1_bn/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:14:00.546932: W tf_adapter/util/infershape_util.cc:337] The shape of node branch_pool_bn/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:14:00.546980: W tf_adapter/util/infershape_util.cc:337] The shape of node 3x3_1x1_bn/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:14:00.547089: W tf_adapter/util/infershape_util.cc:337] The shape of node branch_1x1_bn/cond/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:14:00.547123: W tf_adapter/util/infershape_util.cc:337] The shape of node 3x3_1x1_bn/cond/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:14:00.547155: W tf_adapter/util/infershape_util.cc:337] The shape of node branch_pool_bn/cond/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:14:00.547303: W tf_adapter/util/infershape_util.cc:337] The shape of node 3x3_1x3_bn/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:14:00.547345: W tf_adapter/util/infershape_util.cc:337] The shape of node 3x3_3x1_bn/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:14:00.547377: W tf_adapter/util/infershape_util.cc:337] The shape of node 3x3_1x3_bn/cond/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:14:00.547414: W tf_adapter/util/infershape_util.cc:337] The shape of node 3x3_3x1_bn/cond/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-10-12 14:14:00.547968: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node group_deps_3 is null.

Evaluated on test-set
SRCC/PLCC: 0.753 0.755
ACCURACY: 0.8170915295062224
```

## 推理/验证过程<a name="section1465595372416"></a>

1. 通过predict_mlsp_wide.py启动测试。

2. 当前只能针对该工程训练出的模型进行推理测试。

3. 测试结束后会打印：被评测图片对应的预测分数

```
predicted image score: 6.3623834
```

## GPU和NPU的精度和性能的对比结果<a name="section1465595372416"></a>
- 训练性能

| |GPU和NPU在执行train_mlsp_wide.py时候的时长对比 |  |  |
|---|---|---|---|
GPU | 20*3个epoch需48h |  |  |
NPU | 20*3个epoch需14h |  |  |

说明：可以看出，NPU上的性能远远超过GPU。

- 训练精度

| |SRCC | PLCC | Accuracy |
|---|---|---|---|
paper | 0.756 | 0.757 | 81.72% |
GPU | 0.752 | 0.754 | 81.55% |
NPU | 0.753 | 0.755 | 81.70% |

说明：可以看出，NPU与GPU精度相当。