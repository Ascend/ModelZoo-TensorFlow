# vat_tf


Tensorflow implementation for reproducing the semi-supervised learning results on CIFAR-10 dataset in the paper "Virtual Adversarial Training: a Regularization Method for Supervised and Semi-Supervised Learning"

论文链接：http://arxiv.org/abs/1704.03976

参考repo: https://github.com/takerum/vat_tf

### Requirements
python 3.7.5
tensorflow-gpu 1.15.0
scipy 0.19.0(for ZCA whitening)

## Preparation of dataset for semi-supervised learning
On CIFAR-10

将CIFAR-10数据集下载到 ./dataset/cifar10 文件夹下。并执行
``` 
cd ./code
python3 cifar10.py --data_dir=../dataset/cifar10/
```

## Semi-supervised Learning without augmentation 
On CIFAR-10

```python3 train_semisup.py --dataset=cifar10 --data_dir=../dataset/cifar10/ --train_url=../checkpoint --num_epochs=500 --epoch_decay_start=460 --epsilon=10.0 --method=vat```

## Evaluation of the trained model
On CIFAR-10

```python3 test.py --dataset=cifar10 --data_dir=../dataset/cifar10/ --train_url=../checkpoint ```

```
精度比对|GPU|NPU  
ACC|0.73|0.75  

性能比对|GPU|NPU  
ms/step|48.71|38.06  
```
