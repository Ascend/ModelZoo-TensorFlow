# 概述
    mean-teacher是一种用于图像分类的半监督学习方法，能够在拥有少量有标签数据的情况下训练出分类准确率很高的网络模型。

- 论文链接: [Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780)

- 官方代码仓: [链接](https://github.com/CuriousAI/mean-teacher/)

- 精度比较

| 推理数据 | Accuracy Train | Accuracy Infer |
| :----:| :----: | :----: |
| 10000 | 85.4%  |  86.8% |

# 环境
推理前请参考下文进行环境配置: [使用ATC转换OM模型](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0004.html) and [使用msame进行推理](https://gitee.com/ascend/tools/tree/master/msame)

# 推理
## 数据准备
    使用cifar10数据集进行训练和推理，数据集路径obs://meanteacher/data/
## 使用data_convert_to_bin.py进行数据预处理
```commandline
#dataset_path： 数据集路径
#bin_path: 预处理后数据和标签的路径
#batch_size: batch大小, 需与OM模型一致，默认1
python data_convert_bin.py --dataset_path=XXX --bin_path=XXX --batch_size=1
```
## OM模型转换命令
```commandline
#参照命令和实际环境设置具体路径和参数
atc --model=/root/mean_acl/pb_model/mean-teacher.pb             \   #pb模型路径
    --framework=3                                               \
    --output=/root/mean_acl/pb_out/mean_teacher                 \   #输出的om模型路径
    --soc_version=Ascend310                                     \
    --input_shape="placeholders/images:1,32,32,3"               \
    --log=info                                                  \
    --out_nodes="output:0"
```

## msame推理命令
```commandline
#参照命令和实际环境设置路径和参数
msame --model /home/HwHiAiUser/mean_teacher_acl/mean_teacher.om \   #用于推理的OM模型路径
      --input /home/HwHiAiUser/mean_teacher_acl/bin/image_1     \   #用于推理的bin格式数据路径
      --output /home/HwHiAiUser/mean_teacher_acl/               \   #推理结果输出路径
      --outfmt TXT
```

## 精度计算
```commandline
python cal_inference_pref.py --LABEL_FLODER=XXX \   #推理结果目录
                             --PREDICT_FLODER=XXX   #label文件路径
```

# 附录
## 推理文件OBS路径：
   - pb模型及转换后om模型: obs://meanteacher/acl/model
   - 转换后的bin数据集：obs://meanteacher/acl/bin/
