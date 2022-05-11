# DD-NET

### 关于项目

本项目为复现"Make Skeleton-based Action Recognition Model Smaller, Faster and Better"论文算法。

论文链接：[paper](https://arxiv.org/pdf/1907.09658.pdf)

原作者开源代码：[code](https://github.com/fandulu/DD-Net)

转换后可在Ascend 910运行的代码：[code](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/DD-NET_ID1088_for_TensorFlow)

### 模型转换

#### h5转pb

使用'tf.keras.models.save_model'将模型保存为h5文件；运行h5topb.py脚本得到固化的pb文件。
 

#### pb转om

使用ATC模型转换工具进行模型转换，命令：
`atc --model=./model.pb --framework=3 --output=./out --soc_version=Ascend310 --input_shape="Input:1,32,105; Input_1:1,32,15,2" --log=info --out_nodes="model_1/dense_2/Softmax:0"`

### 推理

使用msame工具，参考命令：
`/home/HwHiAiUser/AscendProjects/tools/msame/out/msame --model ./out.om --output ./output --outfmt BIN --loop 1
`

