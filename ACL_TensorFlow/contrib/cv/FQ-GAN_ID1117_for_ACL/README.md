## 概述

特征量化（FQ），将真数据样本和假数据样本嵌入到共享离散空间中。FQ的量化值被构造为一个进化词典，与最近分布历史的特征统计一致。因此，FQ隐式地在紧凑空间中实现了鲁棒的特征匹配。我们的方法可以很容易地插入现有的GAN模型中，在训练中几乎没有计算开销。

- 参考论文：

  [Feature Quantization Improves GAN Training](https://arxiv.org/abs/2004.02088)

- 参考实现：[YangNaruto](https://github.com/YangNaruto)/**[FQ-GAN](https://github.com/YangNaruto/FQ-GAN)**

- 适配昇腾 AI 处理器的实现：

  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/FQ-GAN_ID1117_for_TensorFlow

  

## 原始模型

百度网盘分享地址：链接：https://pan.baidu.com/s/1EhnLjVI-abNbpPGV_ptwBA 
提取码：so7q 


步骤一:通过代码ckpt2pb.py将ckpt转成pb

## pb模型

```
UGATIT_new.pb
```

百度网盘地址：链接：https://pan.baidu.com/s/1k7AfDg0aYB5RMHJ9IaGjYw 
提取码：2ndx

## om模型

使用atc命令将pb文件转为om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```shell
atc --model=/home/test_user07/FQ-GAN/FQ-U-GAT-IT/pb_model/UGATIT_new.pb --framework=3 --output=/home/test_user07/FQ-GAN/FQ-U-GAT-IT/pb_model/UGATIT_acc_new --soc_version=Ascend910 --input_shape="input1:1,256,256,3" --log=info --out_nodes="p_cam:0;c_out:0" --precision_mode=force_fp32 --op_select_implmode=high_precision
```

成功转化成UGATIT_acc_new.om

UGATIT_acc_new.om的百度网盘地址：链接：https://pan.baidu.com/s/12TWKBjuem6q4RqUmrru6Jw 
提取码：dxxw 


## 数据集转换bin

使用自己修改过的img2bin.py将jpg格式的测试图片转为bin格式。

命令为：

```shell
python2 /home/test_user07/tools/img2bin/img2bin.py -i/home/test_user07/FQ-GAN/FQ-U-GAT-IT/dataset/selfie2anime/testB -w 256 -h 256 -f RGB -a NHWC -t float32 -c [1,1,1] -o /home/test_user07/FQ-GAN/FQ-U-GAT-IT/dataset/selfie2anime/testB_out_new 
```

转换后的bin文件见百度网盘地址:链接：https://pan.baidu.com/s/15vpEjtbuse9XZ9bqK72kiA 
提取码：wte9

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

使用msame推理工具，参考如下命令，发起推理测试：

```shell
./msame --model "/home/test_user07/FQ-GAN/FQ-U-GAT-IT/pb_model/UGATIT_acc_new.om" --input "/home/test_user07/FQ-GAN/FQ-U-GAT-IT/dataset/selfie2anime/testB_out_new"  --output "/home/test_user07/FQ-GAN/" --outfmt TXT
```

推理后会在目的地址下面生成一个以日期命名的文件夹，每个测试图片对应生成的推理结果以名称对应，如图：![image-20221016213203471](https://gitee.com/hkx888/ModelZoo-TensorFlow/raw/master/ACL_TensorFlow/contrib/cv/FQ-GAN_ID1117_for_ACL/image-20221016213203471.png)

## 推理精度

生成图片的格式为txt格式，通过代码post_pro.py代码处理结果，转化为jpg格式的图片。

```shell
python3 post_pro.py --input="/home/test_user07/FQ-GAN/20221016_18_13_56_414701"
```

使用ckpt测试，生成图片。

```shell
python3.7 main.py --dataset='selfie2anime' --phase='test' --test_train=False --quant=True --epoch=100 --iteration=10000
```

对比结果，数值结果和图片结果均相同：

![image-20221016223754779](https://gitee.com/hkx888/ModelZoo-TensorFlow/raw/master/ACL_TensorFlow/contrib/cv/FQ-GAN_ID1117_for_ACL/image-20221016223754779.png)

ckpt生成：

![711665931223_.pic](https://gitee.com/hkx888/ModelZoo-TensorFlow/raw/master/ACL_TensorFlow/contrib/cv/FQ-GAN_ID1117_for_ACL/ckpt_out.jpg)

推理结果还原：

![721665931223_.pic](https://gitee.com/hkx888/ModelZoo-TensorFlow/raw/master/ACL_TensorFlow/contrib/cv/FQ-GAN_ID1117_for_ACL/msame_out.jpg)



