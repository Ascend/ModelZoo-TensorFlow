

# VAE离线推理

VAE离线推理生成MNIST图片

## 环境

 Python版本：3.7.5

 CANN 版本：5.0.3 

 推理芯片： 昇腾310 


## 快速入门

### 1. 获取数据集

使用MNIST数据集，在下一步预处理中会自动下载。

### 2. 图片预处理

将jpegs图片预处理成bin文件.

```
cd scripts
mkdir input_bins
python3 preprocessing.py
```


### 3. 将pb文件转为om模型

pb文件已和源码一起一起上传至仓库中


将pb文件转为om模型

```
atc --model=xxx/hh.pb --framework=3 --output=xxx/hh_om --soc_version=Ascend310 --input_shape="input:100,784"
```

### 4. 构建推理程序


```
bash build.sh
```

### 5. 进行离线推理
进行离线推理得到输出bin文件，需要修改benchmark_tf.sh中的路径。

```
cd scripts
bash benchmark_tf.sh

```
### 6. 对输出bin文件进行后处理生成MNIST图片


```
python3 postprocess.py ./results/VAE/hh_output0.bin
```

## 参考实现

训练代码
[1] https://github.com/kvfrans/variational-autoencoder
