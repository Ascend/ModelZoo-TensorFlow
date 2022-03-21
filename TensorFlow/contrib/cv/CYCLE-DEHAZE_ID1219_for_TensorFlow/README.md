# Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing

原始模型参考[github链接](https://github.com/engindeniz/Cycle-Dehaze)，本项目将原始模型迁移到NPU。

## Requirments

* TensorFlow 1.15
* Python 3.7.5
* MATLAB 
* Ascend 910
* GPU运行平台: Tesla V100
* NPU运行平台: ModelArts
* 依赖的其他网络: vgg16.npy
* Dataset: NYU-Depth

## 代码路径解释

```
CYCLE-DEHAZE_ID1219_for_TensorFlow
|_
  |-- data                   ----存放数据(.png)
      |-- allData            ----存放数据集中所有图片
          |-- clearImage     ----存放数据集中所有清晰的图片
          |-- hazyImage      ----存放数据集中所有有雾的图片
      |-- trainData          ----存放用于训练的图片，由allData中的图片划分而来
          |-- clearImage   
          |-- hazyImage
      |-- testData           ----存放用于测试的图片，由allData中的图片划分而来
          |-- hazyImage
          |-- groundtruth
          |-- model_predict  ----测试时生成，存放模型输出的去雾图片
      |-- tfrecords          ----存放训练时使用的tfrecords文件
  |-- boot_modelarts.py      ----modelarts平台启动文件
  |-- checkpoints            ----训练时生成，保存ckpt文件
  |-- pretrained             ----训练时生成，保存pb文件
  |-- build_data.py
  |-- cal_metrics.py         
  |-- convertHazy2GT.sh
  |-- discriminator.py 
  |-- export_graph.py
  |-- generator.py
  |-- help_modelarts.py
  |-- inference.py
  |-- laplacian.m
  |-- model.py
  |-- ops.py
  |-- reader.py
  |-- resize_im.m
  |-- shuffle_copy.py
  |-- test.sh                ----离线测试启动脚本
  |-- train_gpu.py    
  |-- train_npu.py         
  |-- train_npu.sh           ----npu训练启动脚本
  |-- train_gpu.sh           ----gpu训练启动脚本
  |-- utils.py
  |-- vgg16.npy              ----需自行下载放在当前位置
  |-- vgg16.py
  |-- LICENSE
  |-- README.md

```

## 数据集准备
&ensp;&ensp;&ensp;&ensp;本项目使用[NYU-Depth](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html#raw_parts)数据集进行训练和测试，此数据集仅包含清晰的图片，进一步生成有雾的图片请参考[Dehaze-GAN](https://www.jianshu.com/p/38911d0f2a29)。
&ensp;&ensp;&ensp;&ensp;数据集的准备过程如下：
1. 将NYU-Depth数据集所有的清晰图片放于data/allData/clearImage目录下，图片命名为0001.png、0002.png...以此类推；将所有对应的有雾图片data/allData/hazyImage目录下。
2. 执行以下命令，从完整数据集中选择指定数量的图片作为训练集和测试集（图片随机选择，训练集和测试集的大小在shuffle_copy.py中可修改），并将训练图片拷贝至data/trainData对应的目录中，将测试图片拷贝至data/testData对应的目录中。
```python
python3 shuffle_copy.py
```
3. 执行以下命令，根据data/trainData中的图片生成tfrecords文件用于训练，tfrecords文件保存于data/tfrecords目录下
```python
python3 build_data.py
```

## Train
### NPU训练
&ensp;&ensp;&ensp;&ensp;将数据集准备阶段生成的tfrecords文件上传到obs供训练使用，然后使用pycharm ModelArts进行训练，ModelArts的使用请参考[模型开发向导_昇腾CANN社区版(5.0.2.alpha005)(训练)_TensorFlow模型迁移和训练_华为云](https://support.huaweicloud.com/tfmigr-cann502alpha5training/atlasmprtg_13_9002.html)
&ensp;&ensp;&ensp;&ensp;配置方式请参考：
<div align="center">
<img src="figure/modelarts_config.png" width="300" height="220">
</div>

### NPU离线推理
&ensp;&ensp;&ensp;&ensp;由于在推理阶段需要MATLAB环境，因此本项目只能支持离线推理。在NPU训练结束后需要将pb文件从obs上下载到特定目录，执行以下命令：
```sh
sh test.sh data/testData/hazyImage data/testData/model_predict models/Hazy2GT.pb
```
&ensp;&ensp;&ensp;&ensp;上述命令的参数设置仅为示例，详细的参数设置请参考脚本中的注释。

### GPU训练
&ensp;&ensp;&ensp;&ensp;本项目主要支持NPU上的训练，若要进行GPU训练则需要对项目代码进行少量的改动，改动如下：
1. 将所有源文件头部导入npu相关模块的代码删除.
2. 将源文件中创建session时config参数设置中与npu相关的语句删除，相当于进行了[官方迁移教程](https://support.huaweicloud.com/tfmigr-cann504alpha2training/atlasmprtg_13_0011.html)的逆过程，示例如下：
   NPU训练时的代码为：
   ```python
   config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
   custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
   custom_op.name = "NpuOptimizer"
   config.graph_options.rewrite_options.remapping = RewriterConfig.OFF 
   with tf.Session(graph=graph, config=npu_config_proto(config_proto=config)) as sess:
   ```
   修改为GPU训练版本之后为：
   ```python
   with tf.Session(graph=graph) as sess:
   ```
3. 将model.py中的代码：
   ```python
        learning_step = (
          npu_distributed_optimizer_wrapper(tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name))
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
   ```
   修改为：
   ```python
        learning_step = (
          tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
   ```
&ensp;&ensp;&ensp;&ensp;修改完成之后执行以下命令，详细的参数设置参考脚本中的注释：
```sh
    sh train_gpu.sh
```
### GPU离线推理
&ensp;&ensp;&ensp;&ensp;gpu训练产生的ckpt文件保存在checkpoints文件夹下，模型pb文件保存在pretrained文件夹下。执行以下命令进行离线推理，详细的参数设置请参考脚本中的注释。
```sh
    sh test.sh data/testData/hazyImage data/testData/model_predict pretrained/Hazy2GT.pb
```

## 迁移结果
### 指标对比
&ensp;&ensp;&ensp;&ensp;GPU和NPU均使用相同的训练集以及测试集，训练参数都相同。作者在论文中提供的各项指标值为：


||PSNR|SSIM|
|:------:|:------:|:------:|
|NYU-Depth|15.41|0.66|



<table align="center">
    <tr>
        <td align="center">metrics</td> 
        <td colspan="2" align="center">PSNR</td> 
        <td colspan="2" align="center">SSIM</td>
   </tr>
   <tr>
        <td align="center">chip</td> 
        <td align="center">gpu</td> 
        <td align="center">npu</td>
        <td align="center">gpu</td>
        <td align="center">npu</td>
   </tr>
    <tr>
        <td align="center">NYU-Depth</td> 
        <td align="center">16.96</td> 
        <td align="center">17.01</td>
        <td align="center">0.61</td>
        <td align="center">0.63</td>    
    </tr>
</table>

### 性能对比
&ensp;&ensp;&ensp;&ensp;仅支持单卡训练，训练性能对比：

|平台|BatchSize|训练性能|
|:------:|:------:|:------:|
|NPU|2|0.29 steps/sec|
|GPU Tesla V100|2|1.49 steps/sec|