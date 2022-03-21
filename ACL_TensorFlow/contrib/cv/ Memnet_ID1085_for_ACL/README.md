# Memnet

### 1.关于项目

本项目为复现"MemNet: A Persistent Memory Network for Image Restoration"论文算法，Memnet模型引入一个由递归单元和门单元组成的记忆块(memory block)，实现了一种非常深层的持久性记忆网络，提高了图像恢复任务的精确度。

论文链接为：[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tai_MemNet_A_Persistent_ICCV_2017_paper.pdf)

原作者开源代码链接为：[code](https://github.com/tyshiwo/MemNet)

tensorflow版本代码链接：[tensorflow-code](https://github.com/lyatdawn/MemNet-Tensorflow)

### 2.关于依赖库

见Requirements.txt，需要安装tensorflow==1.15.0, opencv-python。

### 3.关于数据集

由于原论文中使用的数据集较小，因此采用了两种数据集进行验证：

采用VOC数据集与gpu版本进行精度对比

采用BSD数据集与论文模型进行精度对比

数据集训练前需要做预处理操作，请用户参考tensorflow版本代码链接，将数据集封装为tfrecord格式。

**note**：tfrecord中的每一条数据为一组图片对，包括了一张clean图片和一张noisy图片，这两张图片都是单通道的灰度图片，请注意。

### 4.关于训练

**GPU**训练：bash train.sh

**NPU**训练：由于是在modelart中进行训练的，直接将将boot file设置为modelarts_entry；

若是在NPU的其他环境下运行，需要将modelarts.entry中的data_url和train_url，分别设置为数据集所在目录以及输出文件目录，在运行python modelarts_entry.py。

#### ckpt转pb

在华为云的Ascend310服务器中，toolkit中包含有freeze_graph.py文件，可以将ckpt转换为pb文件，可参考如下命令：

`python3.7 /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/tools/freeze_graph.py `
`--input_checkpoint=/home/checkpoint/model-1` 
`--output_graph=./Memnet.pb --output_node_names="MemNet_M6R6/add_46"` 
`--input_meta_graph=/home/checkpoint/model-1.meta`  

#### pb转om

使用ATC模型转换工具进行模型转换，可参考如下命令：

`atc --model=/memnet.pb --framework=3` 
`--output=/home/tmp/memnet` 
`--soc_version=Ascend310` 
`--input_shape="Placeholder_1:1,256,256,1"` 
`--log=info` 
`--out_nodes="MemNet_M6R6/add_46:0"`

### 5.性能&精度

#### 性能

| GPU V100   | Ascend 910  |
| :--------- | :---------- |
| 0.19s/iter | 0.088s/iter |

#### 训练精度

| GPU V100 | Ascend 910 |
| :------- | :--------- |
| 99.99%   | 99.98%     |

#### 推理精度

|      | GPU V100 | Ascend 910 |
| ---- | -------- | ---------- |
| PSNR | 28.347   | 27.945     |
| SSIM | 0.851    | 0.829      |
