-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** 3D point cloud segmentation 

**版本（Version）：1.1**

**修改时间（Modified） ：2021.11.3**

**大小（Size）：6.91MB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的squeezeseg三维点云实例和语义分割网络训练代码** 

<h2 id="概述.md">概述</h2>

DenseNet-121是一个经典的图像分类网络，主要特点是采用各层两两相互连接的Dense Block结构。为了提升模型的效率，减少参数，采用BN-ReLU-Conv（1*1）-BN-ReLU-Conv（3*3）的bottleneck layer，并用1*1的Conv将Dense Block内各层输入通道数限制为4k（k为各层的输出通道数）。DenseNet能有效缓解梯度消失，促进特征传递和复用。 

- 参考论文：

  [[1710.07368\] SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud (arxiv.org)](https://arxiv.org/abs/1710.07368)

- 参考实现：

    [BichenWuUCB/SqueezeSeg: Implementation of SqueezeSeg, convolutional neural networks for LiDAR point clout segmentation (github.com)](https://github.com/BichenWuUCB/SqueezeSeg)

- 适配昇腾 AI 处理器的实现：

  

  ​    


- 通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理：

  - 输入数据为.npy格式的numpy数据
  - 原始数据读入后进行采样

- 测试数据集预处理：

  - 输入数据为.npy格式的numpy数据
  - 原始数据读入后进行采样

- 训练超参

  - Batch size: 32
  - Momentum: 0.9
  - Learning rate(LR): 0.01
  - Optimizer: MomentumOptimizer
  - Weight decay: 0.0001
  - Decay steps: 10000
  - Train epoches: 100


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否     |
| 混合精度   | 是       |
| 并行数据   | 是       |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
在train.py中添加改行代码：
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  ```


<h2 id="训练环境准备.md">训练环境准备</h2>

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


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备

1. 模型训练使用KITTI数据集，数据集请用户自行获取。

2. 数据集训练前需要做预处理操作，请用户参考[Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim),将数据集封装为tfrecord格式。

3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本train.py中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
     #在train.py中修改FLAGS中的信息：
     
     #将data_path改成对应的数据集存放位置（若在modelarts上运行则无需修改此路径）
     tf.app.flags.DEFINE_string('data_path', '/home/ma-user/modelarts/inputs/data_url_0/', """Root directory of data""")
     
     #将checkpoint_path改成对应的模型存放位置（若在modelarts上运行则无需修改此路径）
     tf.app.flags.DEFINE_string('ckpt_path', '/home/ma-user/modelarts/outputs/train_url_0/',
                                """Directory where to write event logs and checkpoint. """)
     
     #需要训练的最大step数：
     tf.app.flags.DEFINE_integer('max_steps', 30000, """Maximum number of batches to run.""")
     
     #每隔多少个step保存一次模型：
     tf.app.flags.DEFINE_integer('checkpoint_step', 1000, """Number of steps to save summary.""")
     ```
  
- 验证。

- 1. 修改验证脚本参数, 在eval.py中配置checkpoint文件所在路径，请用户根据实际路径进行修改。

     ```
     #在eval.py中修改FLAGS中的信息：
     
     #将data_path改成对应的数据集存放位置（若在modelarts上运行则无需修改此路径）
     tf.app.flags.DEFINE_string('data_path', '/home/ma-user/modelarts/inputs/data_url_0/',
                                """Root directory of data""")
                                
     #将checkpoint_path改成对应的模型存放位置（若在modelarts上运行则无需修改此路径）                           
     tf.app.flags.DEFINE_string('ckpt_path', '/home/ma-user/modelarts/outputs/train_url_0/',
                                """Directory where to write event logs and checkpoint. """)
     
     #run_once设为True:只对模型存放路径中的指定模型做一次测试
     #run_once设为False:对模型存放路径中的最新保存的模型进行测试，同时也不断读取路径中后续保存的最新模型
     tf.app.flags.DEFINE_boolean('run_once', True, """Whether to run eval only once.""")
     
     #若run_once 为True,即只需要测试一个模型时，此时还要再修改具体模型的名称。
     tf.app.flags.DEFINE_boolean('test_model', "model.ckpt-xxxx", """The model which will be tested.""")
     例如tf.app.flags.DEFINE_boolean('test_model', "model.ckpt-1000", """The model which will be tested.""")
     ```
     
  2. 训练及测试指令（./scripts/train_and_eval.sh）
  
   ```
     bash ./scripts/train_and_eval.sh
     或者
     python ./scripts/run_sh.py
   ```

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     如果要使用自己的数据集，需要将数据集放到data_url对应目录下。参考代码中的数据集存放路径如下：

     - 训练集：'s3://squeezeseg-training/SqueezeSeg/data/'
  
     - 测试集：'s3://squeezeseg-training/SqueezeSeg/data/'
  
     - 数据集存放结构：
  
       ```
       ├── data    
       |    ├──ImageSet
       |        ├──all.txt
       |        ├──train.txt
       |        ├──val.txt
       |    ├──SqueezeNet
       |        ├──squeezenet_v1.1.pkl #预训练权重
       |    ├──lidar_2d
       训练和测试时的data_path为：/xxx/xxx/data/
       ```
  
  2. 准确标注类别标签的数据集。
  
  3. 数据集每个类别所占比例大致相同。

- 模型修改

  1. 模型分类类别修改。 

     1.1 使用自有数据集进行分类，如需将分类类别及类别数进行修改，修改src/config /kitti_squeezeSeg_config.py

         mc.CLASSES = ['unknown', 'car', 'pedestrian', 'cyclist']
         mc.NUM_CLASS = len(mc.CLASSES)
         同时还要修改相应的mc.CLS_COLOR_MAP
         mc.CLS_COLOR_MAP= np.array([[0.00, 0.00, 0.00],
                                     [0.12, 0.56, 0.37],
                                     [0.66, 0.55, 0.71],
                                     [0.58, 0.72, 0.88]])


- 加载预训练模型。 

  1.配置文件参数，修改文件train.py，增加以下参数：


    tf.app.flags.DEFINE_string('pretrained_model_path', './data/SqueezeNet/squeezenet_v1.1.pkl',
                               """Path to the pretrained model.""")

​      2.修改src/config/config.py中的配置内容：

```
cfg.LOAD_PRETRAINED_MODEL = True
```

- 模型训练。

  参考“模型训练”中训练步骤。

- 模型评估。

  参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── README.md                               //代码说明文档
├── scripts
│    ├──run_sh.py                          //启动训练脚本
│    ├─train_and_eval.sh                   //启动测试脚本
├── src
|    ├──config 
|        ├──config.py                       //配置文件
|        ├──kitti_squeezeSeg_config.py      //配置文件
|    ├──imdb
|        ├──imdb.py                         //读取原始数据并分割生成训练所需数据
|        ├──kitti.py                        //生成训练集以及验证集数据文件名
|    ├──nets
|        ├──squeezeSeg.py                   //网络结构文件
|    ├──utils
|        ├──util.py
|    ├──demo.py
|    ├──eval.py                             //测试文件
|    ├──nn_skeleton.py                      //网络结构及损失函数
|    ├──train.py                            //训练文件
|    ├──fusion_switch.cfg                   //用于关闭modelarts上的融合算子。
```

## 脚本参数<a name="section6669162441511"></a>

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动网络训练。

2. 参考脚本的模型存储路径

3. NPU训练过程打屏信息如下，训练过程中，NPU性能优于GPU训练性能：

4. GPU训练性能（0.173s/batch）：

   ```
   2021-11-03 16:53:47.682725: step 2, loss=2.70 (183.9 images/sec; 0.174 sec/batch)
   2021-11-03 16:53:47.856994: step 3, loss=3.12 (183.7 images/sec; 0.174 sec/batch)
   2021-11-03 16:53:48.028846: step 4, loss=2.98 (186.3 images/sec; 0.172 sec/batch)
   2021-11-03 16:53:48.201724: step 5, loss=3.28 (185.2 imaqes/sec; 0.173 sec/batch)
   2021-11-03 16:53:48.374140: step 6, loss=2.81 (185.7 images/sec; 0.172 sec/batch)
   2021-11-03 16:53:48.546630: step 7, loss=2.42 (185.6 images/sec; 0.172 sec/batch)
   2021-11-03 16:53:48.719973: step 8, loss=2.41 (184.7 images/sec; 0.173 sec/batch)
   2021-11-03 16:53:48.892851: step 9, loss=3.12 (185.2 images/sec; 0.173 sec/batch)
   2021-11-03 16:53:49.065666: step 10, loss=2.18 (185.3 images/sec; 0.173 sec/batch)
   ```

5. NPU训练性能（0.144s/batch）：

   ```
   2021-11-26 16:35:35.374193: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:765] The model has been compiled on the Ascend AI processor, current graph id is:11
   2021-11-26 16:35:37.972771: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:765] The model has been compiled on the Ascend AI processor, current graph id is:21
   2021-11-26 16:37:12.734711: step 0, loss = 3.07 (0.3 images/sec; 97.303 sec/batch)
   2021-11-26 16:37:13.044775: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:765] The model has been compiled on the Ascend AI processor, current graph id is:31
   2021-11-26 16:37:22.672808: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:765] The model has been compiled on the Ascend AI processor, current graph id is:41
   2021-11-26 16:37:35.051020: step 1, loss = 3.34 (2.2 images/sec; 14.700 sec/batch)
   2021-11-26 16:37:35.194878: step 2, loss = 3.17 (222.8 images/sec; 0.144 sec/batch)
   2021-11-26 16:37:35.337124: step 3, loss = 2.72 (225.1 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:35.478714: step 4, loss = 3.06 (226.1 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:35.620530: step 5, loss = 3.27 (225.8 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:35.762009: step 6, loss = 3.41 (226.3 images/sec; 0.141 sec/batch)
   2021-11-26 16:37:35.904036: step 7, loss = 3.72 (225.4 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:36.046333: step 8, loss = 3.23 (225.0 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:36.188445: step 9, loss = 3.29 (225.3 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:36.330835: step 10, loss = 3.13 (224.9 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:36.473366: step 11, loss = 2.69 (224.6 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:36.615527: step 12, loss = 3.33 (225.2 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:36.758108: step 13, loss = 2.84 (224.6 images/sec; 0.143 sec/batch)
   2021-11-26 16:37:36.900358: step 14, loss = 2.44 (225.1 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:37.042039: step 15, loss = 2.92 (226.0 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:37.184430: step 16, loss = 2.30 (224.9 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:37.329822: step 17, loss = 2.52 (220.3 images/sec; 0.145 sec/batch)
   2021-11-26 16:37:37.472960: step 18, loss = 1.95 (223.7 images/sec; 0.143 sec/batch)
   2021-11-26 16:37:37.615536: step 19, loss = 1.84 (224.9 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:37.757993: step 20, loss = 1.94 (224.8 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:37.900911: step 21, loss = 2.20 (224.1 images/sec; 0.143 sec/batch)
   2021-11-26 16:37:38.043715: step 22, loss = 2.15 (224.3 images/sec; 0.143 sec/batch)
   2021-11-26 16:37:38.186252: step 23, loss = 2.54 (224.7 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:38.329071: step 24, loss = 1.53 (224.2 images/sec; 0.143 sec/batch)
   2021-11-26 16:37:38.471636: step 25, loss = 1.97 (224.6 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:38.614355: step 26, loss = 2.14 (224.5 images/sec; 0.143 sec/batch)
   2021-11-26 16:37:38.757248: step 27, loss = 2.07 (224.1 images/sec; 0.143 sec/batch)
   2021-11-26 16:37:38.900075: step 28, loss = 1.89 (224.2 images/sec; 0.143 sec/batch)
   2021-11-26 16:37:39.043122: step 29, loss = 1.86 (223.9 images/sec; 0.143 sec/batch)
   2021-11-26 16:37:39.185984: step 30, loss = 2.58 (224.1 images/sec; 0.143 sec/batch)
   2021-11-26 16:37:39.328827: step 31, loss = 1.72 (224.2 images/sec; 0.143 sec/batch)
   2021-11-26 16:37:39.471274: step 32, loss = 2.03 (224.8 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:39.613228: step 33, loss = 2.75 (225.6 images/sec; 0.142 sec/batch)
   2021-11-26 16:37:39.755284: step 34, loss = 3.57 (225.5 images/sec; 0.142 sec/batch)
   
   ```

   6.NPU训练精度略低于GPU训练精度，尤其在cyclist类。

   NPU训练的模型平均测试精度：

   ```
   Accuracy:
       car:
   	Pixel-seg: P: 0.628, R: 0.973, IoU: 0.617
       pedestrian:
   	Pixel-seg: P: 0.364, R: 0.292, IoU: 0.193
       cyclist:
   	Pixel-seg: P: 0.249, R: 0.498, IoU: 0.199
   ```

   GPU训练的模型平均测试精度：

   ```
   Accuracy:
       car:
   	Pixel-seg: P: 0.637, R: 0.973, IoU: 0.626
       pedestrian:
   	Pixel-seg: P: 0.371, R: 0.353, IoU: 0.221
       cyclist:
   	Pixel-seg: P: 0.270, R: 0.613, IoU: 0.231
   ```

## 推理/验证过程<a name="section1465595372416"></a>

1.  通过“模型训练”中的测试指令启动测试。

2.  当前只能针对该工程训练出的checkpoint进行推理测试。

3.  推理脚本的参数eval_dir可以配置为checkpoint所在的文件夹路径，则该路径下所有.ckpt文件都会根据进行推理。

4.  测试结束后会打印验证集的top1 accuracy和top5 accuracy，如下所示。

```

```