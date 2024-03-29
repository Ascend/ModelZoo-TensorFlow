# **BlitzNet_ID0948_for_Tensorflow**

## 目录

-   [基本信息](#基本信息)
-   [概述](#概述)
-   [训练环境准备](#训练环境准备)
-   [快速上手](#快速上手)
-   [高级参考](#高级参考)

## 基本信息

**发布者（Publisher）：Huawei
**应用领域（Application Domain）： CV
**版本（Version）：1.1
**修改时间（Modified） ：2021.8.26
**大小（Size）：242K
**框架（Framework）：TensorFlow 1.15.0
**模型格式（Model Format）：ckpt
**精度（Precision）：Mixed
**处理器（Processor）：昇腾910
**应用级别（Categories）：Research
**描述（Description）：联合执行对象检测和语义分割，实现实时计算

<h2 id="概述">概述</h2>

BlitzNet在一次前向传递中联合执行对象检测和语义分割，从而实现实时计算。同时表明对象检测和语义分割在准确性方面起相互促进作用。


- 参考论文：

    http://arxiv.org/abs/1708.02813v1  

- 参考实现：

  https://github.com/dvornikita/blitznet

- 适配昇腾 AI 处理器的实现：

  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/BlitzNet_ID0948_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 默认配置<a name="section91661242121611"></a>

- 数据集预处理（以voc2012训练集为例，仅作为用户参考示例）：

  请参考“概述”->“参考实现”

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 并行数据   | 否       |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  run_config = NPURunConfig(        
  		model_dir=flags_obj.model_dir,        
  		session_config=session_config,        
  		keep_checkpoint_max=5,        
  		save_checkpoints_steps=5000,        
  		enable_data_pre_proc=True,        
  		iterations_per_loop=iterations_per_loop,        			
  		log_step_count_steps=iterations_per_loop,        
  		precision_mode='allow_mix_precision',        
  		hcom_parallel=True      
        )



<h2 id="训练环境准备">训练环境准备</h2>

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
   <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.2</em></p>
   </td>
   <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">5.0.2</a></em></p>
   </td>
   </tr>
   </tbody>
   </table>


<h2 id="快速上手">快速上手</h2>

- 训练数据集准备
  
  OBS下载地址：（下载的数据集为处理完的tf数据集）
  https://blitznets.obs.myhuaweicloud.com:443/Datasets/voc12-train-seg?AccessKeyId=UC40X3U4Z2RUPSTV8ADH&Expires=1661686224&Signature=QkWct66ZOwIUfNOYeoWFFZ/FTsk%3D
  
- ResNet预训练模型准备

  OBS下载地址：（将下载的resnet50_full.ckpt文件置于Weights_imagenet中）
  https://blitznets.obs.myhuaweicloud.com:443/resnet50_full.ckpt?AccessKeyId=UC40X3U4Z2RUPSTV8ADH&Expires=1661686362&Signature=P3jAkJ63oqyuneHBq/qAglvS3ts%3D
  

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 单卡性能训练。

     用户可以执行test/train_performance_1p.sh脚本执行少量step获取性能信息：

     ```
     cd test
     bash train_performance_1p.sh --data_path=数据集路径   

     train_performance_1p.sh中调用的训练命令示例如下：
     python3 train_1p.py --obs_dir=${obs_url} --run_name=BlitzNet300_x4_VOC12_detsegaug --dataset=voc12-train --trunk=resnet50 --x4 --batch_size=16 --optimizer=adam --detect --segment --max_iterations=10 --lr_decay 25000 35000
     ```

  2. 单卡精度训练。

     用户可以执行test/train_full_1p.sh脚本执行少量step获取性能信息：

     ```
     cd test
     bash train_full_1p.sh --data_path=数据集路径   

     train_performance_1p.sh中调用的训练命令示例如下：
     python3 train_1p.py --obs_dir=${obs_url} --run_name=BlitzNet300_x4_VOC12_detsegaug --dataset=voc12-train --trunk=resnet50 --x4 --batch_size=16 --optimizer=adam --detect --segment --max_iterations=40000 --lr_decay 25000 35000

  3. 执行结果。    

  |精度指标项|论文发布|GPU实测|NPU实测|
  |---|---|---|---|
  |ACC|xxx|0.88|0.88|

  |性能指标项|论文发布|GPU实测|NPU实测|
  |---|---|---|---|
  |FPS|XXX|0.35 sec/batch|0.23 sec/batch|


<h2 id="高级参考">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── Weights_imagenet                          //用于存放Resnet预训练模型
├── README.md                                 //代码说明文档
├── train_1p.py                               //训练代码
├── resnet.py                                 //resnet模型处理
├── resnet_utils.py                           //resnet模型处理
├── resnet_v1.py                              //resnet模型处理
├── scripts
│    ├──train_testcase.sh                    //自测试用例脚本
```

