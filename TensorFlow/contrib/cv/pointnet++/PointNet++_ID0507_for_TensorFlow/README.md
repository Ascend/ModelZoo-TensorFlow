-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Point cloud Segmentation、Point cloud Classification 

**版本（Version）：1.2**

**修改时间（Modified） ：2021.09.27**

**大小（Size）：52.3M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的PointNet++点云分割网络训练代码** 

<h2 id="概述.md">概述</h2>

PointNet++是一个点云特征提取网络，可用于点云分割和点云分类任务中。PointNet++利用所在空间的距离度量将点集划分为有重叠的局部区域。在此基础上，首先在小范围中利用PointNet提取局部特征，然后扩大范围，在这些局部特征的基础上提取更高层次的特征，直到提取到整个点集的全局特征。

- 参考论文：

    [Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas. “PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
.” 	arXiv:1706.02413](https://arxiv.org/pdf/1706.02413.pdf) 

- 模型的Tensorflow实现（官方开源Tensorflow GPU版）
    [https://github.com/charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)

- 适配昇腾 AI 处理器的实现
    [https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/pointnet++/PointNet++_ID0507_for_TensorFlow
    ](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/pointnet++/PointNet++_ID0507_for_TensorFlow)   
    
- 通过Git获取对听commit_id的代码方法如下：
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```  

<h2 id="默认配置.md">默认配置</h2>

- 训练数据集预处理（当前代码以ShapeNetPart数据集为例，仅作为用户参考实例）
    - 点云的输入尺寸为2048*3
    - 点云采样方法为随机采样
 
- 训练超参（单卡）
    - num_point: 2048
    - batch_size: 8
    - learning_rate: 0.001
    - momentum: 0.9
    - decay_step: 200000
    - optimizer: adam
    - decay_rate: 0.7
    - max_epoch: 201

<h2 id="支持特性.md">支持特性</h2>

| 特性列表 | 是否支持 |
|  :----:   | :----: |
| 分布式训练 | 否 |
| 混合精度 | 是 |
| 数据并行 | 否 |

<h2 id="混合精度训练.md">混合精度训练</h2>
昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

<h2 id="开启混合精度.md">开启混合精度</h2>

相关代码示例

```
sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["dynamic_input"].b = 1
custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
```

<h2 id="训练环境准备.md">训练环境准备</h2>

  1.硬件环境准备请参见[各硬件产品文档](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909) 需要在硬件设备上安装CANN版本配套的固件与驱动。
  
  2.宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/index) 获取镜像。
    
   当前模型支持的镜像列表如表1所示。
    
   表1镜像列表
    
   |镜像名称|镜像版本|配套CANN版本|
   | :----: | :----: | :----: |
   | ARM架构: [ascend-tensorflow-arm](https://ascendhub.huawei.com/#/detail/ascend-tensorflow-arm)| 21.0.2 | 5.0.2|
   | x86架构: [ascend-tensorflow-x86](https://ascendhub.huawei.com/#/detail/ascend-tensorflow-x86)|  |  |
    
<h2 id="快速上手.md">快速上手</h2>

<h3 id="数据集准备.md">数据集准备</h3>

1.模型训练使用ShapeNetPart数据集，ShapeNetPart数据集包含16个物体，总共16862个点云数据，并且每个数据中对应的点都标记了对应的类别。

2.数据集请用户在官网自行获取shapenetcore_partanno_segmentation_benchmark_v0_normal.zip，或通过如下链接获取：
    - [ShapeNetPart Dataset](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)

3.数据集下载后，放在相应目录下，在训练脚本中制定数据集路径，可正常使用。

<h2 id=模型训练.md>模型训练</h2>

 - 下载工程代码，并选择合适的下载方式下载源码包
 
 - 启动训练前，首先要撇值程序运行相关环境变量，环境变量配置信息参见：
 
   [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
   
 - 单卡训练
 
   - 配置训练参数
   
     首先在part_seg/command.sh中配置训练数据集路径等参数，请用户根据实际路径配置
   
   ```
    NA
   ```
   
   - 启动单卡训练（脚本位于PointNet++_ID0507_for_Tensorflow/scripts/train_1p.sh）

   ```
     bash train_1p.sh
   ```
   
 - 验证
    - 测试的时候需要修改脚本启动参数（脚本位于PointNet++_ID0507_for_Tensorflow/scripts/train_1p.sh）
    ```
    ```
    - 启动脚本（脚本位于PointNet++_ID0507_for_Tensorflow/scripts/train_1p.sh）
    ```
      bash test_1p.sh
    ```
  
 - 在线推理
 
    - 配置推理参数
    
        需要配置数据集的根目录--dataset_path，和固化的pb模型路径--model_path
    
    - 启动在线推理
    
    ```
      python3 infer_from_pb.py
    ```
        
 - 生成pb模型
 
    - 执行ckpt转pb的冻结脚本，请用户根据实际路径配置参数
    
    ```
      python3 frozen.py --model_path part_seg/checkoints/model.ckpt --output_model_path ./pb_model/pointnet2.pb
    ```
   [pb模型，提取码：111111](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=cb6s/tGUosqPz4V5C7qhUtm/gdOiGFcPf8za1t3fx8B9RRk6XvXHEO25EfSUXicIkuyAP3+c6UB2tzjKN5gi9BgfRPGG2+EtKIvR9HlY2Bid2es1xcZYzTrCRJiB8BHc/zn9AZ1OOZ9bIKR3S0vs0RSjK6GmKHoXYzLUL9ArXR4JDD/dPuf97KVjgTct3sOpYTECM2oq/JlLmmeVG62uaTTFRc/brbHMQnhfABwsy5yvBvBQujJwjw27eQAGA1YgXVBR5RJAFR9Zo75SSibyUUUp4QN7BikknZ0l7zI2+r8UalAUXP2mm90REChplch5Z/vpbKGYgFI/l/VWJyTGsyCy+D8sLbzMTnxEA3AKOJAwFcNIHrizPsQCSUdXS8VTn/m1Irlw8SUwabSRTyXXoxKoqbwSSiTlwyyfquNA1J9cFPvS+dmwGfELSpLIFW8Nq7qJHwfacr6pRlC+N6wH77+qBkkhMlFpMI4nmi2aX+2FiDhDpfivSbp2573Z8NqNl2oSqyd5geJxV6J0sgK35waM6o6vtmU9XnbmN+KvbEs=)
   
  - 转om模型
  
    - 以refer_test模式为例，使用atc命令，即可得到对应的om模型：
    
    ```
      atc --model=./pb_model/pointnet2.pb --framework=3 --output=./pointnet2 --soc_version=Ascend310 --input_shape='input:1,2048,6'
    ```
 
 - NPU与GPU训练速度对比
 
    -  NPU使用ASCEND910，96GB显存，GPU使用V100，12GB显存。TensorFlow版本为1.15。Batch_size=8情况下，训练速度对比：

  |    GPU    |    NPU    |
  | :-------: | :-------: |
  | 0.5s/iter | 3.8s/iter |
    
 - NPU与GPU训练精度对比
 
    - 在ShapeNetPart数据集上抽取了2880个点云数据作为测试数据，通过IoU量化评价指标评测模型精度。IoU为点云分割中分割正确的点数与点云中所有点的比值，IoU越大，分割精度越高。
    
    - 在相同超参下，模型训练相同的迭代次数（200次），训练精度对比如下：
    
      ```
        NPU上模型的推理IoU精度为：84.26
        GPU上模型的推理IoU精度为：84.32
      ```
    
    [精度数据(提取码：111111)](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=cb6s/tGUosqPz4V5C7qhUtm/gdOiGFcPf8za1t3fx8B9RRk6XvXHEO25EfSUXicIkuyAP3+c6UB2tzjKN5gi9BgfRPGG2+EtKIvR9HlY2Bid2es1xcZYzTrCRJiB8BHc/zn9AZ1OOZ9bIKR3S0vs0RSjK6GmKHoXYzLUL9ArXR4JDD/dPuf97KVjgTct3sOpYTECM2oq/JlLmmeVG62uaTTFRc/brbHMQnhfABwsy5z+ilQLMoIyEM+BVWiZ7GKIp1ZWcOOLcV+LyjWcfMneidUhAZFJp2APizHycGhY6VNEdA1yAUmrB0QnexK4g/PKqJ5w1EbnCRwg4aKdAmX3/8LHpVKj1WXRd6dspiVXeJ7dSDJhPks1XisKwHd8TDlEKMRTvijdccm35/e2CBtw+ZnEAW+T2Ywyyq692mpmWR7oqBSgzohTivcmg/AHUZVA1yiohb3tay2ln/w+QzBDtnPsChrgtsPnHzp3qBGTvk0qh7QmgGeKR4Xuy7Pnvyz21bVk0buRFskmG1zmb/Gq0W8Na/h3DDc93F6GjH/Ivtd7+PTD/QHRfcJeT4SUP6i0XiVaBoIDq87W3+fJCfzXyreJtHCiiVoljR4JU5g1rEA=
    )
<h2 id="高级参考.md">高级参考</h2>

<h3 id="脚本和示例代码.md">脚本和示例代码</h3>

    ```
    |-evaluate.py                            //物体分类网络测试代码
    |-LICENES                                //license文件
    |-modelnet_dataset.py                    //modelnet数据集加载代码
    |-modelnet_h5_dataset.py                 //从h5格式文件加载modelnet数据集代码
    |-modelzoo_level.txt                     //模型指标完成情况文件
    |-README.md                              //说明文档
    |-requirements.txt                       //环境配置文件
    |-train.py                               //物体分类训练代码
    |-train_multi_gpu.py                     //多GPU上物体分类训练代码
    |-data
    |   |-README.md                          //数据集说明文件
    |-doc
    |   |-teaser.jpg                         //PointNet++网络架构图
    |-models
    |   |-pointnet2_cls_msg.py               //多尺度物体分类代码
    |   |-pointnet2_cls_ssg.py               //单尺度物体分类代码
    |   |-pointnet2_part_seg.py              //零件分割模型代码
    |   |-pointnet2_part_seg_msg_one_hot.py  //多尺度零件分割模型代码文件，以one hot形式编码
    |   |-pointnet2_sem_seg.py               //点云分割代文件
    |   |-pointnet_cls_basic.py              //物体分类代码
    |-part_seg
    |   |-evaluate.py                        //测试代码
    |   |-part_dataset.py                    //数据集加载代码
    |   |-part_dataset_all_normal.py         //数据集加载代码，包含点云的法向量
    |   |-infer_from_pb.py                   //在线推理代码
    |   |-frozen.py                          //模型固化代码
    |   |-train.py                           //训练代码
    |   |-train_one_hot.py                   //训练代码，以one hot形式编码
    |-scannet
    |   |-preprocessing
    |   |   |-collect_scannet_scenes.py      //ScanNet数据集预处理文件
    |   |   |-demo.py                        //数据预处理测试demo
    |   |   |-fetch_label_names.py           //解析annotation文件，解析点云数据场景中所有类别
    |   |   |-scannet-labels.combined.tsv    //点云数据集类别annotation文件
    |   |   |-scannet_util.py                //ScanNet数据集处理工具函数
    |   |-pc_util.py                     //点云和体素相互转换工具函数代码
    |   |-README.md                      //ScanNet数据集处理说明文件
    |   |-scannet_dataset.py             //ScanNet数据集加载文件
    |   |-scene_util.py                  //ScanNet数据集格式转换文件
    |   |-train.py                       //ScanNet数据集分类训练代码
    |-scripts
    |   |-test_1p.sh                     //单卡测试脚本
    |   |-train_1p.sh                    //单卡训练脚本
    |-tf_ops
    |   |-3d_interpolation
    |   |   |-tf_interpolating.py        //点云插值代码
    |   |-grouping
    |   |   |-tf_group.py                //点云分组代码
    |   |-sampling
    |   |   |-tf_sample.py               //点云采样代码
    |-utils
    |   |-compile_reder_balls_so.sh      //点云可视化.so库编译脚本
    |   |-pc_util.ps                     //点云和体素相互转化代码
    |   |-pointnet_util.py               //PointNet++网络层定义代码
    |   |-provider.py                    //数据处理相关的代码
    |   |-README.md                      //点云可视化工具说明文档
    |   |-render_balls_so.cpp            //点云可视化库代码
    |   |-show3d_balls.py                //点云可视化代码
    |   |-tf_util.py                     //Tensorflow网络层实现代码
    ```
    
<h3 id="脚本参数.md">脚本参数</h3>
 
    ```
    --model                         model file
    --log_dir                       the path of log
    --num_point                     the number of pointcloud to resample
    --max_epoch                     epoch for training
    --batch_size                    batch size for training
    --learning rate                 initial learning rate
    --momentum                      momentum for training
    --optimizer                     optimizer for training adam or momentum
    --decay_step                    decay step for learning rate decay
    --decay_rate                    decay rate for learning rate decay
    ```
    
<h3 id="训练过程.md">训练过程</h3>
 
NA

<h3 id="推理/验证过程.md">推理/验证过程</h3>

NA