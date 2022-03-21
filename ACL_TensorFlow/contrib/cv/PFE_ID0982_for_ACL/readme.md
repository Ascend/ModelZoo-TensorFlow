# README

## 1、关于项目

本项目将PFE迁移到ascend 910进行训练得到的模型在ascend 310上进行离线推理。

相比传统的将人脸图像确定性嵌入到隐空间的一个点，Probabilistic Face Embeddings(PFEs)提出在隐空间建立人脸图像的分布，这样可以在无约束的人脸识别环境下有更好的表现。而且可以基于传统的确定性嵌入模型作为预训练模型。在此基础上训练PFE模型。

![PFE](../../../../TensorFlow/Research/cv/pfe/pfe_ID0982_for_TensorFlow/PFE.png)

论文： [paper](http://arxiv.org/abs/1904.09658)

论文源代码： [code](http://github.com/seasonSH/Probabilistic-Face-Embeddings)

## 2、关于依赖库

见requirements.txt。

## 3、关于测试集

测试集采用了使用LFW数据集，并进行预处理成112*96，放在data/lfw_mtcnncaffe_aligned下，后续模型在推理过程中，先获得每个图片Gaussian分布的mu和sigma，再根据lfw_pairs.txt中的图片对，进行匹配推理。

obs测试数据集： [obs_lfw](http://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=qwwgvJK2FYxY566/N/OLIaNhRd6sMlvWfOd8dAmFgEjGPacPPn75WjtvHmk7Rn9UWMFLvTPCFjYsA1bLZhNJFjTSDEio+2PWSN7aBpMQFRrRAoBTjzQTDyEEy3P1HwSChco4zQVHRuyT7m/s6BkX34C76LFUbCcBaa2/0WLzxiyswxCQgC5MIVAmjsjuAUKZH23o3BKLjNO89jXCovZq0kVIh9Z+V8P9npbmKq7GTcLFg5GTxspBjAF7FncxogVXRKpuwCadGj41Vpo/mVMgM/zikysjbddlp8zHi4BXNloGmJuZ+mkPSiCpOS74AxRskpuQqVSGGEe2OZOZQ50WuT+kCSgOSum4PtfSaEdM2GMKWYUlYuNK5CM+zlDmF+SKc6vlboEOca4bXUluPMW85NChqLCoi8Fh6jAeA183NrPIuo9o7gSr8pEXP8PirJA8RyYkW4aLD1T9qDCY9Qs7u8pRTkhlVSze0hplh6Mne/rtZ6vJOL0ih0WYWs0iFCBpvvC8HDerT+wHGNTfApytT//G5eNwdH7vqhYOjMtJaHAb8nVFaS6bHvo3yKar4AdHZwRiGuy1RM5I7YNwU2TCjpdZadjUJPtR2KNTD/u1eoM=
) ，提取码： yzy123

## 4、pb模型

原始ckpt文件下载后，在ckpt2pb.py中修改相应的ckpt文件路径（使用ckpt-3000）。执行以下命令转换为pb模型。

```
python3 ckpt2pb.py
```
ckpt2pb.py中固定了pb模型的输出目录为./pb_model/。

ckpt的obs地址：[obs_ckpt](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=qwwgvJK2FYxY566/N/OLIcBv9MNQUw5xiowx1FhY10go0NerrJQi7DQxxLVuolM9zrjiTL/hbCp4D+N5Jo2LJ3B3Edbbn/LqtSapramHmAsgunJG1e5v7rkNBYCf+Aj10io6lkapYhDjtA8ZqxwbxD3m3tKshv9h+0h5rWf4iH2YUffRMoHkRVjTVHyTQvkpmDpZmTovmzj0tav7YfpTswyFiLk3Qa+dVgz3BlsBv8vwBy4elYrJaj601kvK7lRqkvMrumnb+B0JUzJg5XPOkjJPMiyg3vEiB9FLG3+D03m1dHKKeZYnAqd/d3TKvgolhS/zghxEz1vq0nxFbeglA4FkI0bPpdXhkV17W6bNVbQ7XjlDn49LJVua1gCl//i9pLh1Yv3fP0ziqEusU8D/PEGQYp9t5sMriR1vrc8iezu49CBLzATqveQSJSSIJOdENRwjnE94QgAVFkNSrsXgHdBEZgdWCUMapfrkaHhxQnIuITkUyreXNETfE5fNLzSJo7i/+RiGXes9bewJCG8Iyvrskb41+Fs6qOINF1rDD9a9b2OHCi/98HLKvZO2ouPYmOYWkCr/okiml3mnNZnFAq6Rc0BfNHjN5UbzDVudIYm5Ifsz4Q+fs/e0xWCiRtbolVCYXWdbI8W4TMQ0WdwAFQ==)，提取码： yzy123

## 5、生成om模型。

使用atc命令将pb模型转换为om模型，执行以下命令转换为om模型。

```
sh pb2om.sh
```

pb模型的obs地址：[obs_pb](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=qwwgvJK2FYxY566/N/OLIaNhRd6sMlvWfOd8dAmFgEjGPacPPn75WjtvHmk7Rn9UWMFLvTPCFjYsA1bLZhNJFjTSDEio+2PWSN7aBpMQFRrRAoBTjzQTDyEEy3P1HwSChco4zQVHRuyT7m/s6BkX34C76LFUbCcBaa2/0WLzxiyswxCQgC5MIVAmjsjuAUKZd8nMKPh5yAZ0lIrzbk2vHquzkU+qGmsVQ9wgg7rq7SAe2PzLNEhxj40FA4j3A70QGUgITX/YTVcclBRP0xTQXXDWX7kZAPUq0rfOwyriNKeGpM0EFQ4Dho2GiUcM17QhZk3GKrYdyUGxJ/Z0D0KkGx+elzRGQFNd7oIcuihGWE5z27gInXw6BmQcstqbh0gT9piXzTNt5MrNzNBkt1wSfaq+JbTpudswlK2KRWNX1Ub67BvXUhLX5w9PFl59bdRsTgIDFE57qygQMK5a80aJeGEgtZj6mGEOdnOz0D+cwmBMf2XzjTGVegx1lY48omYO7BeA10tRj/yAvY840omR5w==)，提取码： yzy123

注意配置对应的文件路径

## 6、测试集内文件转换为bin文件

执行以下命令将测试集内的jpg文件经过处理转换为用于网络输入的bin文件。inf_input2bin.py文件中固定了生成bin文件存放的目录为./input_bin。

```
python3 inf_input2bin.py
```

## 7、使用om模型进行推理

使用msame工具进行推理。参考命令如下。

```
sh om_inf.sh
```

注意配置对应的文件路径


## 8、om模型离线推理性能

推理的平均运行性能为18.62ms。


## 9、om模型离线推理的accuracy值

在om_output_inf.py中修改推理输出的bin文件的路径。执行以下命令查看推理生成的bin文件的accuracy值。

```
python3 om_output_inf.py
```
||Euclidean (cosine) accuracy|MLS accuracy|
|--|--|--|
|base|0.99233|0.99383|
|om|0.99233|0.99133|

