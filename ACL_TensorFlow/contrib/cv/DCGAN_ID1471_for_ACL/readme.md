### 推理部分
参照原论文，将 DCGAN 的鉴别网络参数存储，数据集 Cifar-10 利用鉴别网络提取特征保存到本地，对保存的特征文件进行 SVM 分类，最终得到分类的精确度为本论文的精度指标。

```shell
python main.py --dataset materials --data_dir=./data/cifar10/ --input_height=64 --crop --train_svm True
```

classifier_svm.py说明

利用参数train_svm控制两个过程

train_svm为True，加载鉴别网络(net_D)，将cifar-10数据集输入网络得到特征，存入文件中(fname = 'cifar10_svm')

train_svm为False，载入特征文件，训练SVM(157-160行注释第一次训练时放开完成训练，得到SVM参数)
加载SVM参数进行predict，最后得到分类结果即为精度

