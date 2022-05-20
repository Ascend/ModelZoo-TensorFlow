# COMPARE_GAN离线推理

## 环境要求

| 环境 | 版本 |
| --- | --- |
| CANN | <5.0.3 |
| 处理器| Ascend310/Ascend910 |
| 其他| 见 'requirements.txt' |

## 数据集bin准备：
    生成64*128的随机噪声
    python noise_bin_data.py
    数据集链接：[OBS](obs://cann-id2103/inference/bin_data/)
## 脚本和示例代码

```text
├── README.md                                //代码说明文档
├── noise_bin_data.py                        //bin数据生成
├── ckpt_to_pb.py                            //模型固化
├── requirements.txt                         //环境依赖
├── LICENSE.txt                              //证书
├── scripts
│    ├──pb_to_om.sh                          //pb转om
├── ckpt                                     //存放ckpt模型
├── pb                                       //存放pb模型
├── bin_data                                 //存放bin文件                          
```

## 模型文件

包括ckpt、pb、om模型文件

下载链接：[OBS](obs://cann-id2103/inference/)

## ckpt文件转pb模型

```bash
python ckpt_to_pb.py
```

## pb模型转om模型

检查环境中ATC工具环境变量，设置完成后，修改PB和OM文件路径PB_PATH和OM_PATH，运行pb_to_om.sh

```bash
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

PB_PATH=/root
OM_PATH=/root

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$PB_PATH/gan.pb --framework=3 \
        --input_shape="split_1:64,128" \
	    --output=$OM_PATH/ganom --soc_version=Ascend310 \
        --out_nodes="generator/Sigmoid:0" \
	    --log=debug

```
## 使用msame工具推理

安装好[msame]([tools: Ascend tools - Gitee.com](https://gitee.com/ascend/tools/tree/master/msame))

```bash
./msame --model /root/ganom.om --input /root/infer_bin_data.bin --output /root/bin_data --outfmt TXT
```
注意，msame生成的推理文件夹是根据时间命名的，类似于20220429_170719这样的格式，需要自己检查路径。
