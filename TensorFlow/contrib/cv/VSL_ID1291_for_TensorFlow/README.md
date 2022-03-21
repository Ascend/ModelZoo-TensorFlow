## Introduction

VSL是一种变分形状学习者，这是一种生成模型，可以在无监督的方式下学习体素化的三维形状的底层结构，通过使用`skip-connections`，该模型可以成功地学习和推断一个潜在的、层次的对象表示。此外，VSL模型还可以很容易的生成逼真的三维物体。该生成模型可以从2D图像进行端到端训练，以执行单图像3D模型检索。实验结果表明，改进后的算法具有定量和定性两方面的优点。 

- 参考论文：

    [Learning a Hierarchical Latent-Variable Model of 3D Shapes](https://arxiv.org/abs/1705.05994) 
    
    对于更详细的结果，可以参考[项目主页](https://shikun.io/projects/variational-shape-learner)

## Requirements
运行`VSL`模型需安装以下依赖：
- matplotlib 1.5
- vtk 8.2.0
- mayavi 4.7.1
- scikit-learn 0.20
- tensorflow-gpu 1.15


在GPU上训练VSL模型时，依赖`mayavi`和`vtk`应下载对应版本，否则`mayavi`容易下载失败，模型运行报错，建议通过[链接](https://www.lfd.uci.edu/~gohlke/pythonlibs/#mayavi) 依次下载相应版本的`PyQt4`、`traits`、`vtk`、`mayavi`的`whl`文件，并通过`pip`命令进行安装。在NPU上训练VSL模型时，由于没有对应`aarch64`架构的`vtk`版本，因此将模型中的与`mayavi`模块相关的代码注释掉。
## Dataset
`VSL`模型使用 [ModelNet](http://modelnet.cs.princeton.edu/) 和 [PASCAL 3D+ v1.0](http://cvgl.stanford.edu/projects/pascal3d.html) 进行训练。 ModelNet 用于一般 3D 形状学习，包括形状生成、插值和分类。 PASCAL 3D 仅用于图像重建。

通过以下链接下载预处理好的数据集: [[link]](https://www.dropbox.com/sh/ba350678f7pbwx8/AAC8-2X1p4BiOKlyYuuxFcDBa?dl=0).

该数据集包括文件 `ModelNet10_res30_raw.mat` 和 `ModelNet40_res30_raw.mat` 是对ModelNet10/40 数据集的体素化。 `PASCAL3D.mat` 表示与图像对齐的体素化 PASCAL3D+。

`ModelNet10_res30_raw.mat` 和 `ModelNet40_res30_raw.mat`都包含一个 `train` 和 `test`。

`PASCAL3D.mat` 包含 `image_train`, `model_train`, `image_test`, `model_test`四个分类。

## Parameters
如果要使用作者预训练好的模型参数，请通过 [链接](https://www.dropbox.com/s/pz5kqi8guq0jxgm/parameters.zip?dl=0) 进行下载。

## Training VSL
请提前通过提供的链接下载 `dataset` 和 `parameters`。如果从零开始训练，则无需使用`parameters`。

`vsl_main.py` 用于一般 3D 形状生成和形状分类实验。 `vsl_imrec.py` 用于3D重建和检索实验。可以分别执行这两个`.py`文件来进行模型的训练。为了正确使用预训练模型的超参数并与论文中的其他实验设置保持一致，请将超参数定义如下：

| |ModelNet40 | ModelNet10 | PASCAL3D (jointly) | PASCAL3D (separately)|
|---|---|---|---|---|
`global_latent_dim` | 10 | 10|10|5|
`local_latent_dim` | 5 | 5|5|2|
`local_latent_num` | 5 | 5|5|3|
`batch_size` | 100 | 100 | 40 | 5|

## Transfer learning

- 使用与GPU训练相同的数据集，请提前通过提供的链接下载 `dataset`。

- 模型修改

  通过使用npu自动迁移工具进行模型的迁移，详细过程请参考[链接](https://support.huaweicloud.com/tfmigr-cann503alpha1training/atlasmprtgtool_13_0006.html)

- 配置启动文件`boot_modelarts.py`,启动训练时，需设置好`train_url` 和 `data_url` 两个路径，详情请参考[链接](https://support.huaweicloud.com/tfmigr-cann503alpha1training/atlasmprtgma_13_0004.html) 。通过修改以下命令中的`python_command`来选择训练`vsl_main.py` 或者`vsl_imrec.py`：

  ```
    ## start to train on Modelarts platform
    python_command = "python %s/vsl_main.py" % code_dir
    print('python command:', python_command)
    os.system(python_command)
  ```

## Reference

脚本和示例代码

```
├── vsl_main.py                                 //Shape Classication代码
├── vsl_imrec.py                               //Shape Reconstruction代码
├── vsl.py                                     //用于Shape Classication的网络模型
├── vsl_rec.py                                 //用于Shape Reconstruction的网络模型
├── README.md                                  //代码说明文件
├── requirements.txt                           //模型依赖
├── boot_modelarts.py                          //在modelarts平台上训练的启动文件
├── help_modelarts.py                          //在modelarts平台与obs之间进行数据传输
├── freeze_graph.py                            //模型固化脚本
├── dataset                                    //数据集文件
│    ├──ModelNet10_res30_raw.mat              //ModelNet10数据集
│    ├──ModelNet40_res30_raw.mat              //ModelNet40数据集
│    ├──PASCAL3D.mat                          //Pascal3D数据集
├── LICENSE
```


说明：当执行vsl_main.py文件在ModelNet40数据集上训练时，选用的超参数`global_latent_dim`、`local_latent_dim`、`local_latent_num`、`batch_size`分别为10、5、5、100，而在ModelNet10数据集上训练时，选用的超参数分别为10、5、5、100；当执行vsl_imrec.py文件在Pascal3D数据集上训练时，选用的超参数分别为10、5、5、40（jointly）以及5,、2、3、5（separately）。


## Result
- 训练性能

| |ModelNet40 | ModelNet10 | PASCAL3D |
|---|---|---|---|
GPU | 500个epoch需22h | 500个epoch需12h | 10个epoch需30min |
NPU | 500个epoch需10h | 500个epoch需4h | 10个epoch需15min |

说明：表中时间为训练所需的平均时间，可以看出，NPU上的性能远远超过GPU。

- 训练精度

| |ModelNet40 | ModelNet10 | PASCAL3D |
|---|---|---|---|
paper | 84.5% | 91.0% | 0.631 |
GPU | 81.38% | 94.69% | 0.6189 |
NPU | 84.49% | 95.15% | 0.6498 |

说明：表中ModelNet40和ModelNet10数据集为`Shape Classfication`精度，其值越大，说明分类精度越好；PASCAL3D数据集为平均`IOU`值，表示形状之间的相似程度，其值越大，说明重建形状与真实形状相似性越好。可以看出，NPU与GPU精度相当。
