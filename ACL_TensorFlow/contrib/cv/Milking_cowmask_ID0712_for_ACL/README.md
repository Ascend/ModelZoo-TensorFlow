# 概述
    milking_cowmask是一种用于图像分类的半监督学习方法，能够在拥有少量有标签数据的情况下训练出分类准确率很高的网络模型。
    本项目在应用了Shake-Shake正则化的26层的Wide-Resnet的基础网络上应用cowmask,得到了很好的分类效果
    
- 论文链接: [Milking CowMask for Semi-Supervised Image Classification](https://arxiv.org/abs/2003.12022)

- 官方代码(JAX): [链接](https://github.com/google-research/google-research/tree/master/milking_cowmask)

- tensorflow在NPU训练代码OBS路径: obs://milking-cowmask/Milking_cowmask_ID0712_for_TensorFlow/

- 精度比较

| 推理数据 | Accuracy Train | Accuracy Infer |
| :----:| :----: | :----: |
| 10000 | 93.21%  |  93.22% |

# 环境
推理前请参考下文进行环境配置: [使用ATC转换OM模型](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0004.html) and [使用msame进行推理](https://gitee.com/ascend/tools/tree/master/msame)

# 推理
## 数据准备
    使用cifar10数据集进行训练和推理，数据集支持自动下载或obs://milking-cowmask/dataset/。
## 使用data_convert_to_bin.py进行数据预处理
```commandline
#dataset_path： 数据集路径
#bin_path: 预处理后数据和标签的路径
#batch_size: batch大小, 需与OM模型一致
python data_convert_bin.py --dataset_path=XXX --bin_path=XXX --batch_size=1
```
## OM模型转换命令
```commandline
#参照命令和实际环境设置具体路径和参数
atc --model=/home/HwHiAiUser/cowmask/milking_cowmask.pb \   #pb模型路径
    --framework=3 \
    --output=/home/HwHiAiUser/cowmask/milking_cowmask \     #转换后OM模型输出路径
    --soc_version=Ascend310 \
    --input_shape="val_x:-1,32,32,3" \
    --log=error \
    --out_nodes="output:0" \
    --dynamic_batch_size="1,4,8,16,32" \
    --precision_mode=force_fp16
```

## msame推理命令
```commandline
#参照命令和实际环境设置路径和参数
msame --masme_path=/home/HwHiAiUser/msame/tools/msame/out \
      --model=/home/HwHiAiUser/cowmask/milking_cowmask.om \     #用于推理的OM模型路径
      --input=/home/HwHiAiUser/cowmask/bin/image_1 \            #用于推理的bin格式数据路径
      --output=/home/HwHiAiUser/msame/out/ \                    #推理结果输出路径
      --dymBatch=1                                              #batch大小
```

## 精度计算
```commandline
python cal_inference_pref.py --LABEL_FLODER=XXX \   #推理结果目录
                             --PREDICT_FLODER=XXX   #label文件路径
```

# 附录
## 推理文件OBS路径：
   - pb模型: obs://milking-cowmask/Milking_cowmask_ID0712_for_ACL/Appendix/pb/
   - om模型: obs://milking-cowmask/Milking_cowmask_ID0712_for_ACL/Appendix/om/