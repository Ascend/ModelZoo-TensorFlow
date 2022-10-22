# MixMatch - A Holistic Approach to Semi-Supervised Learning


参考文献: "[MixMatch - A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)" 

参考实现：[https://github.com/google-research/mixmatch](https://github.com/google-research/mixmatch)

适配昇腾AI处理器的实现：
https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/mixmatch_ID0804_for_TensorFlow


## 原始ckpt模型


百度网盘地址：链接：https://pan.baidu.com/s/1Wic_t5NueIWZ4eTmX32gDg?pwd=yel5  提取码：yel5


## pb模型

```bash
运行 ckpt2pb.py将ckpt转为pb
```

生成的pb模型：mixmatch.pb

百度网盘地址：链接：https://pan.baidu.com/s/19RlwmCAHkEqE6fX3yVS7Kg?pwd=y2ne  提取码：y2ne


## om模型


pb转om的命令如下：

```bash
bash ATC_PB_2_OM.sh --model=/home/TestUser03/code/mixmatch/mix_model/mixmatch.pb --output=/home/TestUser03/code/mixmatch/mix_model/mixmatch_om_final  --input_shape="x:1,32,32,3"
```
args：

--model：pb模型的路径

--out：生成的om模型的路径

生成的om模型：mixmatch_om_final.om

百度网盘地址： 链接：https://pan.baidu.com/s/1-acaXTD79IIgygfwWjraBg?pwd=9m10 提取码：9m10


## 数据集转换bin

```bash
运行 convert_img_2_bin.py将数据集转化为bin格式
```

## 使用msame工具推理


参考https://gitee.com/ascend/tools/tree/master/msame， 获取msame推理工具及使用方法。

使用msame推理工具，命令如下，进行推理测试。

```bash 
bash om_inference.sh --msame_path=/home/TestUser03/tools/msame/out --model=/home/TestUser03/code/mixmatch/mix_model/mixmatch_om_final.om --input=/home/TestUser03/code/mixmatch/mix_model/input_bin_01 --output=/home/TestUser03/code/mixmatch/mix_model/out --outfmt=TXT 
```
args：

--msame_path: msame推理工具的路径

--model: om模型的路径

--input：数据集转化bin生成的文件路径

--out：生成文件的路径

--outfmt：生成文件的格式


## 精度对比

```bash
运行 cal_infer_acc.py 计算推理的精度
```
图片数量：10000 推理精度：0.8073 训练精度：0.8735



