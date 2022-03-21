## SRDRM: Underwater Image Super-Resolution using Deep Residual Multipliers
原始模型参考 [github链接](https://github.com/xahidbuffon/srdrm), 迁移训练代码到NPU

### Requirments
- Tensorflow 1.15.0
- Ascend 910
- Tesla V100
- NPU运行平台 ModelArts
- 其他依赖参考requirements.txt
- Dataset: USR-248(请参考https://github.com/xahidbuffon/srdrm自行下载)
- Model: **SRDRM** and **SRDRM-GAN** for underwater image super-resolution 

### 代码路径解释
```shell
|-- checkpoints                                      ----存放训练模型的路径[训练时自动生成]
|   |-- USR_2x
|       |-- srdrm 
|       |-- srdrm-gan 
|   |-- USR_4x
|       |-- srdrm
|       |-- srdrm-gan 
|   |-- USR_8x
|       |-- srdrm
|       |-- srdrm-gan 
|-- image                                            ----存放每500step sample样本路径[训练时自动生成]
|   |-- USR_2x
|       |-- srdrm
|       |-- srdrm-gan 
|   |-- USR_4x
|       |-- srdrm
|       |-- srdrm-gan 
|   |-- USR_8x
|       |-- srdrm
|       |-- srdrm-gan 
|--- nets                                           ----网络结构路径
|    |-- SRDRAM.py                                  ----srdrm-gan
|    |-- gen_models.py                              ----srdrm
|--- scripts                                        ---- 模型训练 测试 评估脚本
|    |--- gpu_scripts                               ---- 使用gpu
|         |-- measure_gpu_1p.sh  
|         |-- test_SR_gpu_1p.sh
|         |-- train_GANs_gpu_1p.sh
|         |-- train_genarative_models_gpu_1p.sh
|    |--- npu_scripts                               ---- 使用npu
|         |-- measure_npu_1p.sh                     ---- NPU下获取单个模型单个epoch结果的SSIM PSNR UQIM指标脚本
|         |-- test_SR_npu_1p.sh
|         |-- train_GANs_npu_1p.sh
|         |-- train_genarative_models_npu_1p.sh
|--- task_log                                       ---- 存放训练与测试控制台输出路径[训练时自动生成]
|    |-- train_[chip]                               ---- chip = [gpu/npu/cpu]
|    |-- test_[chip]_epoch_[xx]                     ---- GPU推理时的log保存路径,[xx]为第 xx epoch 模型推理
|    |-- measure                                    ---- 获取评价指标代码的log路径
|        |-- gpu
|        |-- npu
|        |-- cpu
|--- tensorborad_log                                ---- 存放tensorboard文件路径[训练时自动生成]
|--- utils
|    |-- data_utils.py                              ---- dataloader
|    |-- Logger.py                                  ---- 记录控制台输出的工具脚本
|    |-- loss_utils.py                              ---- 计算loss
|    |-- plot_utils.py                              ---- 绘制图片
|    |-- print_config.py                            ---- 打印任务参数配置
|    |-- ssim_psnr_utils.py                          ---- 计算SSIM PSNR
|    |-- uqim_utils.py                              ---- 计算UIQM
|--- weights                                        ---- 存放VGG19预训练权重[自行下载]
|    |--vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5 
|--- boot_modelarts.py                              ---- modelarts启动文件
|--- help_modelarts.py                              ---- modelarts与obs传输文件
|--- LICENSE
|--- measure.py                                     ---- 获取结果指标的脚本
|--- README.md
|--- requirements.txt                                ---- 依赖配置文件
|--- test_SR.py                                     ---- 模型测试脚本
|--- train_GANs.py                                  ---- srdrm-gan模型训练脚本
|--- train_genarative_models.py                     ---- srdrm模型训练脚本
```

### 数据集准备 & 预训练模型下载
- Dataset: USR-248(请参考https://github.com/xahidbuffon/srdrm自行下载)
```shell
数据集组织
|--- dataset
|   |-- SRDRM
|       |-- USR248
|           |-- TEST
|           |-- train_val
```
- vgg19预训练模型下载 [百度网盘链接, 提取码：qkv3](https://pan.baidu.com/s/16J07ou2tecjX1HIz50Qgnw)  [github下载链接](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5)
下载后放在weights目录下。

### GPU训练
命令行切换路径到`scripts/gpu_scripts`

- 训练srdrm-gan, 详细的参数设置请参考脚本中的注释
```shell
sh train_GANs_gpu_1p.sh
```
- 训练srdrm, 详细的参数设置请参考脚本中的注释
```shell
sh train_genarative_models_gpu_1p.sh
```
### GPU测试
命令行切换路径到`scripts/gpu_scripts`，执行以下命令，详细的参数设置请参考脚本中的注释
```shell
sh test_SR_gpu_1p.sh
```
### GPU测试结果评估

命令行切换路径到`scripts/gpu_scripts`，执行以下命令，详细的参数设置请参考脚本中的注释

```shell
sh measure_gpu_1p.sh
```

### NPU训练
命令行切换路径到`scripts/npu_scripts`

- 训练srdrm-gan, 详细的参数设置请参考脚本中的注释
```shell
sh train_GANs_npu_1p.sh <--code_dir> <--data_dir> <--result_dir> <--obs_url>
```
- 训练srdrm, 详细的参数设置请参考脚本中的注释
```shell
sh train_genarative_models_npu_1p.sh <--code_dir> <--data_dir> <--result_dir> <--obs_url>
```
### NPU测试
命令行切换路径到`scripts/npu_scripts`，执行以下命令，详细的参数设置请参考脚本中的注释
```shell
sh test_SR_npu_1p.sh <--code_dir> <--data_dir> <--result_dir> <--obs_url>
```
### NPU测试结果评估

命令行切换路径到`scripts/npu_scripts`，执行以下命令，详细的参数设置请参考脚本中的注释

```shell
sh measure_npu_1p.sh <--code_dir> <--data_dir> <--result_dir> <--obs_url>
```

### 指标对比
均使用相同的训练集以及测试集，训练参数都相同。

SRDRM NPU/GPU Checkpoints: ([百度云链接，提取码：fxca](https://pan.baidu.com/s/12-FvoO7TqJ43zUFDyROYtg)) 

SRDRM-GAN NPU/GPU Checkpoints: ([百度云链接，提取码：uuik](https://pan.baidu.com/s/12na1B9FIfJcu_rEmhNsotg)) 

作者论文中提供的各项指标值为：训练以USR_8x数据集为例。

|           | PSNR              | SSIM           | UIQM           |
| --------- | ----------------- | -------------- | -------------- |
| SRDRM     | 28.36/24.64/21.20 | 0.80/0.68/0.60 | 2.78/2.46/2.18 |
| SRDRM-GAN | 28.55/24.62/20.25 | 0.81/0.69/0.61 | 2.77/2.48/2.17 |

**(Average PSNR, SSIM, and UIQM scores for 2×/4×/8× SISR on USR-248 test set.)**


##### USR_8X  
<table>
    <tr>
       <td>metrics</td>
       <td colspan="2" align="center">PSNR</td>
       <td colspan="2" align="center">SSIM</td>
       <td colspan="2" align="center">UIQM</td>
    </tr>
    <tr>
      <td>chip</td>
      <td>gpu</td>
      <td>npu</td>
      <td>gpu</td>
      <td>npu</td>
      <td>gpu</td>
      <td>npu</td>
    </tr>
    <tr>
      <td>srdrm</td>
      <td>22.74</td>
      <td>23.70</td>
      <td>0.63</td>
      <td>0.63</td>
      <td>2.28</td>
      <td>2.26</td>
    </tr>
    <tr>
      <td>srdrm-gan</td>
      <td>21.49</td>
      <td>21.40</td>
      <td>0.58</td>
      <td>0.58</td>
      <td>2.97</td>
      <td>2.95</td>
    </tr>
</table>



### 性能对比

此处只展示srdrm与srdrm-gan模型在USR_8X数据集上的训练与测试结果。 完整log文件，请参考[百度网盘 提取码：hdmw](https://pan.baidu.com/s/1CsHpas23vPQDC0WqOJepyQ)

#### 训练性能

###### 1.SRDRM for USR_8x

 NPU性能log截图

![image-20211013102523123](https://gitee.com/windclub/image_bed/raw/master/img/20211013102523.png)

 GPU性能log截图

![image-20211013102301982](https://gitee.com/windclub/image_bed/raw/master/img/20211013102309.png)

|   平台   | BatchSize | 训练性能(fps) |
| :------: | :-------: | :-----------: |
|   NPU    |     2     |     11.6      |
| GPU V100 |     2     |     7.71      |

srdrm模型NPU训练性能约为GPU训练性能的**1.5**倍。

###### 2.SRDRM-GAN for USR_8x

NPU性能log截图 

![image-20211128182123865](https://gitee.com/windclub/image_bed/raw/master/img/20211128182124.png)

GPU性能log截图

![image-20211128183855551](https://gitee.com/windclub/image_bed/raw/master/img/20211128183855.png)

|   平台   | BatchSize | 训练性能(fps) |
| :------: | :-------: | :------: |
|   NPU    | 2 | 4.33 |
| GPU V100 |     2     |    2.56    |

srdrm-gan模型NPU训练性能约为GPU训练性能的**1.69**倍。



