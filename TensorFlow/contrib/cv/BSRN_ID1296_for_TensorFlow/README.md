## BSRN: Lightweight and Efficient Image Super-Resolution with Block State-based Recursive Network

原始模型参考 [github链接](https://github.com/idearibosome/tf-bsrn-sr/), 迁移训练代码到NPU

### Requirments
- Tensorflow 1.15.0
- Ascend 910
- GPU运行平台 Tesla V100
- NPU运行平台 ModelArts
- 其他依赖参考requirment.txt
- Dataset: 
  - 训练数据集：DIV2K 
  - 验证数据集：BSD100

- Model: **BSRN** for Super Resolution(超分辨率)

### 代码路径解释
```shell
├── temp										----存放训练结果及数据集文件
│   ├── result									----存放训练结果（gpu训练自动生成）
│   │   ├── model.ckpt							----存放固化的模型pbtxt文件
│   │   ├── result-pictures						----存放验证数据（运行validate_gpu.sh）自动生成的超分辨率图片
│   │   │   ├── ensemble
│   │   │   │   ├── x2
│   │   │   │   ├── x3
│   │   │   │   └── x4
│   └── dataset									----数据集文件
│       ├── BSD100								----验证数据集
│       │   ├── LR
│       │   │   ├── x2
│       │   │   ├── x3
│       │   │   └── x4
│       │   └── SR
│       └── DIV2K								----训练数据集
│           ├── DIV2K_train_HR
│           └── DIV2K_train_LR_bicubic
│               ├── X2
│               ├── X3
│               └── X4
├── tf-bsrn-sr
│   ├── checkpoints								----原始代码提供的训练好的模型文件，用作精度和性能比较
│   ├── dataloaders								----数据预处理和加载脚本，可以得到batch-size大小的数据
│   ├── models									----模型网络定义，保存，恢复及优化相关脚本
│   ├── scripts									----存放模型训练和验证脚本
│    	├── run_gpu.sh							----使用gpu(v100)
│    	├── run_npu.sh							----使用npu(modelarts)
│    	├── run_npu_restore.sh					----从中断点恢复npu训练
│    	├── test.sh								----推理
│    	├── validate_gpu.sh						----gpu(v100)上验证模型精度
│    	└── validate_npu.sh						----npu(modelarts)上验证模型精度
│   ├── boot_modelarts.py
│   ├── help_modelarts.py
│   ├── test_bsrn.py							----测试模型
│   ├── train.py								----训练模型
│   ├── output.txt								----训练输出(gpu训练自动生成)
│   └── validate_bsrn.py						----验证模型
├── statics										----存放图片静态数据(用于md文件)
├── LICENSE
├── README.md
└── requirments.txt  							---- 依赖配置文件

```

### 数据集准备 
- Dataset: (请参考(https://github.com/idearibosome/tf-bsrn-sr/自行下载)
  - 训练数据集：DIV2K 
  - 验证数据集：BSD100
```shell
数据集组织
├── dataset									----数据集文件
    ├── BSD100								----验证数据集
    │   ├── LR
    │   │   ├── x2
    │   │   ├── x3
    │   │   └── x4
    │   └── SR
    └── DIV2K								----训练数据集
        ├── DIV2K_train_HR
        └── DIV2K_train_LR_bicubic
            ├── X2
            ├── X3
            └── X4
```
### GPU训练
命令行切换路径到`tf-bsrn-sr/`

- 训练bsrn, 详细的参数设置请参考脚本中的注释
```shell
nohup bash scripts/run_gpu.sh > output.txt 2>&1 &
```
- 训练之前需修改`boot_modelarts.py`中第77行代码为bash_header = os.path.join(code_dir, 'scripts/run_gpu.sh')

### GPU离线推理<font color='red'> 【在线推理待完善】 </font>

命令行切换路径到`tf-bsrn-sr/`，执行以下命令，详细的参数设置请参考脚本中的注释
```shell
bash scripts/test.sh
```
### GPU评估

命令行切换路径到`tf-bsrn-sr/`，执行以下命令，详细的参数设置请参考脚本中的注释

```shell
bash scripts/validate_npu.sh
```

### NPU训练、推理、评估

使用pycharm ModelArts进行训练

ModelArts的使用请参考[模型开发向导_昇腾CANN社区版(5.0.2.alpha005)(训练)_TensorFlow模型迁移和训练_华为云 (huaweicloud.com)](https://support.huaweicloud.com/tfmigr-cann502alpha5training/atlasmprtg_13_9002.html)

配置方式请参考：

<img src="statics\modelarts配置.PNG" alt="modelarts配置" style="zoom: 67%;" />

（修改`boot_modelarts.py`中第77行代码bash_header = os.path.join(code_dir, 'scripts/run_npu.sh')，可以设置在NPU上跑还是在GPU上跑）

### 指标对比
均使用相同的训练集以及测试集，训练参数都相同。

NPU Checkpoints: ([百度云链接，提取码：xxxx]()) <font color='red'> 【链接待完善】 </font>

GPU Checkpoints: ([百度云链接，提取码：xxxx]()) <font color='red'> 【链接待完善】 </font>

作者论文中提供的各项指标值为：

|      | PSNR   | SSIM   |
| ---- | ------ | ------ |
| BSRN | 27.538 | 0.7341 |

**(PSNR, SSIMscores for scale x4 on BSD100 dataset.)**


##### *×*4-scale BSRN model <font color='red'> 【bsrn gpu, npu指标 待完善】 </font>
<table>
    <tr>
       <td>metrics</td>
       <td colspan="2" align="center">PSNR</td>
       <td colspan="2" align="center">SSIM</td>
    </tr>
    <tr>
      <td>chip</td>
      <td>gpu</td>
      <td>npu</td>
      <td>gpu</td>
      <td>npu</td>
    </tr>
    <tr>
      <td>BSRN</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
</table>




### 性能对比

展示bsrn模型在DIV2K 数据集上的训练性能

#### 训练性能

###### 1.BSRN

 NPU性能log截图

<img src="statics\NPU性能.jpg" alt="NPU性能" style="zoom:67%;" />

 GPU性能log截图

<img src="statics\GPU性能.jpg" alt="NPU性能" style="zoom:67%;" />

|   平台   | BatchSize | 训练性能(sec/batch) |
| :------: | :-------: | :-----------------: |
|   NPU    |     8     |        0.739        |
| GPU V100 |     8     |        0.828        |

#### 推理性能 <font color='red'> 【待完善】 </font>

NPU性能log截图

GPU性能log截图



|   平台   | BatchSize | 训练性能(fps) |
| :------: | :-------: | :-----------: |
|   NPU    |           |               |
| GPU V100 |           |               |

#### 性能调优 <font color='red'> 【待完善】 </font>

##### NPU AutoTune性能

训练时开启AutoTune:

npu训练性能（命令行截图）

| 平台 | BatchSize | 训练性能(fps) |
| :--: | :-------: | :-----------: |
| NPU  |           |               |
