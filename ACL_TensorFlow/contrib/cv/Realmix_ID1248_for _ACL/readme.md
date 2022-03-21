推理情况表   
| 模型 |数据集| 输入shape | 输出shape | 推理时长(单张) | msame精度 | 目标精度 |
|--|--|--|---| -- | --| -- |
| Realmix | CIFAR10 val  | `1*32*32*3` | `1*10`  | 0.88ms~ | 91% | 89.5% | 

## 1、原始模型
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=rNiMW+tUA+0C/QagOP8ihwzWXjGA7KOxhbccfKEu7liirwdE7jd2Kxx0/ssTqbhskWaxexRz/17gwIWZKMLm+8okIEIoeUgT0OLSyPnedoKULaD6XKXfVnJ7AKTJBT0wmq4xl8D5T00nSm+UkN8gPOngDo0qOUZbd2NMTblbS41H9u4HP2T9SsUmBn412IGZC8AlR0V9qYgxMBnKJTK9lCdv0asFgQss8d2HDaxCmd7XtnQWEWEu9Xo1vqPprHJ32w55D5ixKhKYy1Kf5YQjBbN8VuGWcw32UxXclKDAaEUzXe2jHaHAVcEpxgHnjiJ9ZWdjpVYip88dczzMhNKHGq32+lGu7SI3j/FE4pOlgue6VMYMxvspPLI/wherhn4ZmNax1BSGAEre30JEkn4uYZ8+kFThG71fVFBAcqJV4Pdv78xFFET+KwITPe9AKwNmNR+w9o2Vll1+yUkDv0qwiOX//xUlHHsqrZjvfPbJeTE4kF/BPsSzSus2ivtJOkon9RV/G3hFvUJEaWZQ2g9UB/n69dkZsdYxSkGjcd+lZSE=

提取码:
123456

*有效期至: 2022/10/27 22:37:10 GMT+08:00

文件夹中有pb文件，使用pb文件进行推理的准确率为91.03%，平均推理时间为7.3ms
![pb推理截图](https://images.gitee.com/uploads/images/2021/1101/224347_a7f4702c_9846121.png "pdinter.png")

## 2、转om模型

atc转换命令参考：

```sh
atc --input_shape="x_p:1,32,32,3" --check_report=/root/realmix/network_analysis.report --input_format=NHWC --output="/root/realmix/realmix_model_1024" --soc_version=Ascend310 --framework=3 --model="/root/AscendProjects/realmix_1024.pb"
```

## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具


## 4、全量数据集精度测试：

测试集总共10000张图片，每一张图片一个bin，压缩后的文件夹存储在以下桶里
https://realmixdata.obs.cn-north-4.myhuaweicloud.com:443/bindata/bins.zip?AccessKeyId=D4TOTDL9MDVCQI7BCCJC&Expires=1666945114&Signature=gJt1KXDxVkPCkpgJV4rqMHsYD1g%3D

### 4.1 执行推理和精度计算
使用以下命令进行推理
```
./msame --model /root/realmix/realmix_model_1024.om" --input "/root/AscendProjects/bins" --output "/root/AscendProjects/output" --outfmt TXT
```
推理结束后使用以下命令验证准确率
```
python3 eval_accuracy.py
```
om推理精度：
![输入图片说明](https://images.gitee.com/uploads/images/2021/1101/224659_3e30b784_9846121.png "acc.png")
om推理时间
![输入图片说明](https://images.gitee.com/uploads/images/2021/1101/224715_352a2a0a_9846121.png "time.png")
可以看到om推理结果和pb推理结果相差小于1%，同时速度提升将近10倍