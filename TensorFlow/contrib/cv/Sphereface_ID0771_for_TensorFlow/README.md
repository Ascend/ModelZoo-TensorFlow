# 概述
    本项目是Deep Hypersphere Embedding for Face Recognition (CVPR 2017)的快速实现。 文章提出了angular softmax（A-Softmax）loss，使卷积神经网络 (CNN) 能够学习角度判别特征以满足最大类间距离和最小类内距离。
    脚本是使用第三方代码和MNIST数据集,已反馈澄清。

- 论文链接: [Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)

- GPU实现(使用MNIST数据集): [链接](https://github.com/YunYang1994/SphereFace)

- 精度性能比较

| 平台 | BatchSize | FPS | s/step | accuracy |
| ------ | ------ | ------ | ------ | ------ |
| v100 | 256 | 8533 | 0.03 | 98.96% |
| Ascend910 | 256 | 12800 | 0.02 | 98.95% |

# 环境
    - python 3.7.5
    - Tensorflow 1.15
    - Ascend 910

# 训练
## 数据集
    配置数据集路径,默认为:./MNIST_data/,已上传于obs://sphere-face/MNIST_data/
##训练超参见train.py参数列表
## 单卡训练命令
```commandline
bash ./test/train_full_1p.sh
```

# 功能测试
少量step(单epoch)运行
```commandline
bash ./test/train_performance_1p.sh
```
# 模型固化
准备checkpoint,默认为 ./ckpt/sphereface.ckpt
```commandline
python3 freeze_graph.py
```

# 部分脚本和示例代码
```text
├── README.md                                //说明文档
├── requirements.txt			//依赖
├──test		                         //训练脚本目录								 
│    ├──train_performance_1p.sh			 
│    ├──train_full_1p.sh
├──train.py                 	     //训练脚本
├──freeze_graph.py              //固化脚本
```

# 输出
模型存储路径为test/output/ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。loss信息在文件test/output/{ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。 
模型固化输出为pb_model/sphereface.pb
## 预训练结果obs路径:
   - checkpoint: obs://sphere-face/SphereFace_ID0771_for_TensorFlow/ckpt/
   - pb模型: obs://sphere-face/SphereFace_ID0771_for_TensorFlow/pb_model/