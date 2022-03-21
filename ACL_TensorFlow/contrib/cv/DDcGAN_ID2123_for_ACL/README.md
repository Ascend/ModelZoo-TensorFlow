# DDcGAN-master

#### 简介：
DDcGAN网络，一种用于多分辨率图像融合的双鉴别器条件生成对抗网络
通过双重鉴别器条件生成对抗性网络，用于融合不同分辨率的红外和可见光图像。支持融合不同分辨率的源图像，在公开数据集上进行的定性和定量实验表明在视觉效果和定量指标方面都优于最新技术。

#### 离线推理

####  1.pb模型   ckpt转pb 
执行bash om_run.sh 生成pb模型。
ckpt模型 链接：URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=/KNQwLR+Bi7vBesJoDxY34zFVN9oXpKVfS/B+NrNjbgtlZTK2xIEx1fPAgaen/XE0nYCmpd5jgnkXzJ/FvcLDTrNdGSHAvuYGTJM5GNox9naxooOpa6T0kT9OFRa8pSJ71//x2mniTnxSI2ED85vdkJ6b0kmUMIzpjFNmT4HK1I7fkIGZRYd2Q2I+DofqdFMWZ7EKsPMr9+WcxLb0dopvDq/luCNy5S2QGDl7WOG8ciNuD4ffyQVoQVyReGffWNCxvFSku4fYaIsJolQdru0UY58O6zuegx963uWBw5fFdexzNNF6L+eSrMVtoypKG7x/SKobINsK1hmroA90/hm7ra91+BAYurKvMAXzv/rFjESm2MZT26WzuDPhaob96SZa/uAeoAq1k3uvd/RaFBCfq635op6SjJrC262VTN2xa792ICodWq6ujEj0urs544h9nyKfEElumwANH+D7vqtj2N/cTRob3IAZSKZegqnMbf5PHaaWVh3dFMKACkcWrBn1lgTAiYALaVgASwFlULcDQ==

提取码:
123456

*有效期至: 2022/12/04 14:34:35 GMT+08:00

生成的pb模型 链接：URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=/KNQwLR+Bi7vBesJoDxY34zFVN9oXpKVfS/B+NrNjbgtlZTK2xIEx1fPAgaen/XE0nYCmpd5jgnkXzJ/FvcLDTrNdGSHAvuYGTJM5GNox9naxooOpa6T0kT9OFRa8pSJ71//x2mniTnxSI2ED85vdkJ6b0kmUMIzpjFNmT4HK1I7fkIGZRYd2Q2I+DofqdFMWZ7EKsPMr9+WcxLb0dopvA4YKB4hW0Z+/+Tmy/MXZu+nd2pYKbeCP8QOCsGQ53fq/dwL1cgpkhYOuijAlu1I2swJzC+HoQWdlwVX+3gQMjMKh4tQLGeZ81DDgthjLs7hxuVq7h+vR+Vb18KBXkDyWZbdCEuSrl37UryvpzxQPDnMeKEufaA0qyYVfk25yd2jUJL6NoWXd6hWo5pONr//bZY7F0F0X6HfyafksSuOoONh0HjGYIeA40+fGRNME15sTHfTi8iCVCQhshO71C7iA5185y9eZJcGbqPP+6FYIhIFQTLrobzVBmujajzRJbxtT7dlDeHK6UkeYk8gKo2isQ==

提取码:
123456

*有效期至: 2022/12/04 14:35:39 GMT+08:00



 **testImage 链接地址** 

https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=/KNQwLR+Bi7vBesJoDxY34zFVN9oXpKVfS/B+NrNjbgtlZTK2xIEx1fPAgaen/XE0nYCmpd5jgnkXzJ/FvcLDTrNdGSHAvuYGTJM5GNox9naxooOpa6T0kT9OFRa8pSJ71//x2mniTnxSI2ED85vdkJ6b0kmUMIzpjFNmT4HK1I7fkIGZRYd2Q2I+DofqdFMBwP/I8R91Iu7CklCMDXLCItRpevirxr9hjG2Tqjtjz6z3one7Z1sziFJkihNmuE4w0w5lp6FT/MdSyVgyI9qk8ogqRPE4dm83OGP2fi2fimaOpobBVmjBR8wv50wb3t79b9z8yIX/IZ3NkIqwxAl/19i8NF9JUO8TT5o7R5I6IA4/C7kE5vLYZsOXnah+f5JG38VDo8MgWpX8lhZMqqo3ucgQ9p1jp8EFk1alewvPcQwTCv1oXxAhGkvX3bTWOegz2Dq/ykdGd0CQb6+rtOLfsMf0+Se/gJvozEiVCgjdp84hfAiShww5UC9g9gkXhk3tJNZTz4gd5NKifnw5mrGV9nwItbI3jBxKyDa7xO//nY=

提取码:
123456


####  2.数据集转换bin
执行python3.7.5 ModelToBin.py

####  3.离线推理
使用msame推理工具，发起推理测试

./msame --model /home/HwHiAiUser/pb_model/out.om --input /home/HwHiAiUser/bin_model/IR1,/home/HwHiAiUser/bin_model/VIS1 --output ./out

model   输入pb模型路径

input  输入数据集路径

output  推理输出路径 
####  4.推理结果后处理

推理结果为二进制.bin格式数据，需要将其转换为可视的.bmp格式图片。
执行python3.7.5 BinToImage.py


####  5.离线推理精度
执行python3.7.5 compute_metrics.py

参考SSIM\EN两个主要指标作为精度验证。
|      | 论文精度|GPU精度 | NPU精度 |离线推理精度|
|------|--------|--------|--------|--------|
| SSIM | 0.5090 | 0.5425 | 0.5768 |0.5676  |
| EN   | 7.3493 | 7.4121 | 7.4955 |7.5503  |


####  6.离线推理性能
Ascend310 离线推理

NPU图像融合平均时间：6847.4ms

![NPU图像融合平均时间](777.png)

NPU离线推理单个图片耗时:6543.48ms

![NPU离线推理单个图片耗时](6666.png)