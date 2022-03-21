# 模型概述

详情请看**DENSEDEPTH_ID806_for_TensorFlow** README.md *概述*

#  数据集

- nyu_test.zip

## 链接

链接：https://pan.baidu.com/s/14dJZygMgIoiFo05J3ZvfnA 
提取码：4521

# pb模型

详情请看**DENSEDEPTH_ID806_for_TensorFlow** README.md *模型固化*

链接：

链接：https://pan.baidu.com/s/1luaU-L4QoUTgqB3BFHwh9A 
提取码：4521

# om模型

使用ATC模型转换工具进行模型转换时可参考如下指令 atc.sh:

```shell
atc --model=./pd/test.pb --input_shape="input_1:1,480,640,3" --framework=3 --output=./om/test --soc_version=Ascend910A --input_format=NHWC
```

具体参数使用方法请查看官方文档。

链接：

链接：https://pan.baidu.com/s/1MpmNuIhC7iByag-Aq5ig_g 
提取码：4521

# 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

## 数据集转换bin

```shell
python3.7.5 npy2bin.py --nyu_dir ./dataset/nyu_test.zip --output ./result/bin --bs 1
```

将nyu_test.zip中的eigen_test_rgb.npy转换为对应批次的bin文件。会生成两个文件夹，分别存放正常的图片数据以及相对应的镜像图片数据，用于评估模型精度。

参数解释：

```
--nyu_dir nyu_test.zip位置 默认：./dataset/nyu_test.zip
--output 生成的bin文件保存位置 默认：./result/bin 
--bs 每个bin文件包含多少张图片数据 默认：1 注：bs大小应该与om模型的输入批次相同
```

## 推理

可参考如下命令 msame.sh：

```shell
msame --model ./om/test.om --input ./bin/image --output ./result/bin_infer/imager --outfmt BIN
msame --model ./om/test.om --input ./bin/image_flip --output ./result/bin_infer/image_flipr --outfmt BIN
```

注：bin_infer/imager、/bin_infer/image_flipr需要提前创建

## 推理结果后处理

## 测试精度

```shell
python3.7.5 offline_infer_acc.py --bs 1 --bin_dir ./result/bin_infer/image --bin_flip_dir ./result/bin_infer/image_flip --nyu_dir ./dataset/nyu_test.zip
```

测试推理精度

参数解释：

```
--bs 推理批次 默认：1
--bin_dir 使用msame推理后生成的bin文件位置 默认：./bin/image
--bin_flip_dir 使用msame推理后生成的镜像bin文件位置 默认：./bin/image_flip
--nyu_dir 测试数据集位置 默认：./dataset/nyu_test.zip
```

## 推理样例展示

```
python3.7.5 bin2image.py --input ./test.bin --output ./test.png --bs 1
```

展示推理所得的bin文件

参数解释：

```
--input 推理所得的bin文件 默认：./test.bin
--output 生成的展示图片的保存位置 默认：./test.png
--bs 推理所得的bin文件包含的图片批次 默认：1 注：bs应当与bin文件批次大小相同	
```

<img src="https://gitee.com/DatalyOne/picGo-image/raw/master/202109121831853.png" alt="test1" style="zoom: 50%;" />

<img src="https://gitee.com/DatalyOne/picGo-image/raw/master/202109121833413.png" alt="test2" style="zoom:50%;" />



# 代码及路径解释

```
DENSEDEPTH_ID0806_for_ACL 	离线推理文件
├── acl_utils.py
├── atc.sh  				act工具 pb==》om 转换命令
├── bin2image.py 			推理数据后处理：展示推理的结果
├── msame.sh				msame工具：om离线推理命令
├── npy2bin.py				推理数据预处理：将nyu_test.zip中的npy文件转换为bin并进行其他图片预处理
└── offline_infer_acc.py 	评估离线推理模型om
├── dataset						数据集位置		
│   └── nyu_test.zip			评估数据集
└── result						结果保存位置		
    └── ...
```

