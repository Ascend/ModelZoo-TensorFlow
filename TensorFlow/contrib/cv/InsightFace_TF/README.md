- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.11.29**

**大小（Size）：648KB**

**框架（Framework）：TensorFlow-gpu_1.14.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：人脸识别算法**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

InsightFace 是基于 MXNet 框架实现的业界主流人脸识别解决方案。

- 参考论文：
  
  [https://arxiv.org/abs/1801.07698](ArcFace: Additive Angular Margin Loss for Deep Face Recognition)

- 参考实现：

  https://github.com/auroua/InsightFace_TF

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/InsightFace_TF

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - Batch size: 32
    - net_depth：100
    - epochs:200
    - momentum:0.9
    - weight_deacy:5e-4
    - eval_datasets:lfw
    - eval_db_path:/datasets/faces_ms1m_112x112
    - image_size:112,112
    - tfrecords_file_path: ./datasets/tfrecords_webface
    - summary_path:./output/summary
    - ckpt_path:./output/ckpt
    - log_file_path:./output/logs
    - saver_maxkeep: 100
    - buffer_size: 10000
    - log_device_mapping:False
    - summary_interval:300
    - ckpt_interval:10000
    - validate_interval:2000
    - show_info_interval:20

    

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

训练数据集路径
  datasets/tfrecords_webface
测试数据集路径
  datasets/faces_ms1m_112x112


## 模型训练<a name="section715881518135"></a>

- 运行train_nets.py文件
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练

        1.首先在脚本train_nets.py中，配置训练数据集和测试数据集参数如下所示：

             ```

             --tfrecords_file_path=./datasets/tfrecords_webface  --eval_db_path=./datasets/faces_ms1m_112x112 

             ```

        2.启动训练
        
             启动单卡训练 （脚本为train_nets.py） 
        
             ```
             python3 train_nets.py

             ``` 


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--README.md                                                      #说明文档									
|--train_nets.py                                                       #训练代码
|--train_nets_mgpu.py
|--train_nets_mgpu_new.py
|--train_nets_mgpu_new.py
|--verification.py
|--eval_ckpt_file.py						
|--requirements.txt                                               #所需依赖
|--datasets                                                       #训练需要的数据集
|--data                                                           #训练需要的数据集
|       |--eval_data_reader.py
|       |--mx2tfrecords.py                                        #数据集转tfrecords格式脚本
|--figures
|--losses
|       |--face_losses.py                                         #损失函数
|--nets			           	                         
|	|--imagenet_classes.py
|	|--L_Resnet_E_IR.py
|	|--L_Resnet_E_IR_fix_issue9.py
|	|--L_Resnet_E_IR_GBN.py
|	|--L_Resnet_E_IR_MGPU.py
|	|--L_Resnet_E_IR_RBN.py
|	|--nets_utils.py
|	|--networks.py
|	|--resnet.py
|	|--tl_layers_modify.py
|	|--vgg16.py
|	|--vgg19.py
|--output         
|--test
|	|--memory_usage_test.py
|	|--resnet_test_static.py
|	|--test_losses.py
|	|--multiple_gpu_test
|	|	|--test_mgpu_mnist.py  
|	|	|--test_tensorlayer.py
|	|--benchmark
|	|	|--gluon_batchsize_test.py  
|	|	|--mxnet_batchsize_test.py
|	|	|--resnet_slim_benchmark.py 
|	|	|--resnet_tl_benchmark.py 
|	|	|--tensorlayer_batchsize_test.py  
|	|	|--utils_final.py
|	|	|--vgg19_slim_benchmark.py
|	|	|--vgg19_tl_benchmark.py                                            
```



## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。

