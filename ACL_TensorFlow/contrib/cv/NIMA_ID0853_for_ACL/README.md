推理情况表   
| 模型 |数据集| 输入shape | 输出shape | 推理时长(单张) | msame精度 | 目标精度 |
|--|--|--|---| -- | --| -- |
| NIMA | AVAdataset val  | `1*224*224*3` | `1*10`  | 0.99ms~ | 0.517 | 0.51| 

## 1、原始模型
找到对应文件夹,下载对应的`h5文件`，使用该文件夹下面的`h5_pb.py`脚本转成pb模型。(下载链接链接：https://pan.baidu.com/s/1TUk6A8_ztG8gzt4-hDWgCw 
提取码：7e9x) 

## 2、转om模型

atc转换命令参考：

```sh
atc --model=nima.pb  --framework=3 --input_shape="input_1:1,224,224,3" --output=./om_model/nima --out_nodes="dense_1/Softmax:0" --soc_version=Ascend310
```

## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具


## 4、全量数据集精度测试：

测试集总共5000张图片，每一张图片一个bin
使用data_make.py脚本,设定好路径后,
执行 python3 data_make.py \

数据集下载地址
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=2774AwJbDFGJ4Rq2DQvRNJKzAPBqsc4oXfLFA5EV7X/g2fjAMc4zxR1st38W21Ai0JYDosS0F+8YvpzWRorZhcfy1+3hBOE4gQA1IXQ1XcZ8O7gmmYxB+xOivZDucoS6kGbk2cfVMYqSWtLSaHxhx+L19LtoxNLGj87hSppPJiqN4vA7RM1GyXDDfdT1bNgY/0cOtTILOcIyByVK9MRH93Pxhxp+cCOxTJiw4dGnQBgndmJ8DTkIewWgSiGaJo/O5NmkmErToOfuG9a0iDO9ibfjbEgsabcNYsdAEdaCnagt2XqAZvgu/M5byefMS4fOIhZiGOG+i1b2kcN+OqwJfkwrw+k0ufeL6L1HoP0dpVUWeRMBKqv9kbAuZ7+XRu/gnzT4SUeQ4PtApW/IL+MZCPp9dcwizocXS7q73/c2wU0KPg1JA2lvbGSdjmjnHl1XXFbmTAc0XwAaMtcACjSN1+v0tkx8RoTLs2yjBvhhzRd0ix9/V26Yvs+C0Vi/JvmkwDDFErvb72h+97MXLOqC+OYCKmEGHp92jQ3s5Ew4udmFvP20gcoaNAohfWkYlGvMhfW0wWG8u41F6tTXg62n4jmfhFA+sNE/ZGnyHTiHBbs=

提取码:
123456

*有效期至: 2022/09/02 14:42:35 GMT+08:00

注：image将保存在xdata文件夹中 label将保存在ydata文件夹中


### 4.3 执行推理和精度计算
进入到该目录\
![输入图片说明](https://images.gitee.com/uploads/images/2021/0907/142737_eed2d77d_8376014.png "屏幕截图.png")
  
执行命令 `./msame --model /root/nima/om_model/nima.om --input  /root/nima/xdata/ --output /root/nima/ypre`
该命令主要功能就是加载om执行推理\
![输入图片说明](https://images.gitee.com/uploads/images/2021/0907/142834_de6864e2_8376014.png "屏幕截图.png")


最后执行python3  SRCC_compute.py 
得到最后推理精度\
![输入图片说明](https://images.gitee.com/uploads/images/2021/0907/143005_d5dfcf8d_8376014.png "屏幕截图.png")