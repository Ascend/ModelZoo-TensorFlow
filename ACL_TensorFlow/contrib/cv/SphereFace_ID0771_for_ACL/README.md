# 概述
    本项目是Deep Hypersphere Embedding for Face Recognition (CVPR 2017)的快速实现。 文章提出了angular softmax（A-Softmax）loss，使卷积神经网络 (CNN) 能够学习角度判别特征以满足最大类间距离和最小类内距离。
    脚本是使用第三方代码和MNIST数据集,已反馈澄清。

- 论文链接: [Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)

- GPU实现(使用MNIST数据集): [链接](https://github.com/YunYang1994/SphereFace)

- NPU训练代码OBS路径: obs://sphere-face/

- 性能比较

| 推理数据 | Accuracy Train | Accuracy Infer |
| :----:| :----: | :----: |
| 5120 | 98.95%  |  99.00% |

# 环境
推理前请参考下文进行环境配置: [使用ATC转换OM模型](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0004.html) and [使用msame进行推理](https://gitee.com/ascend/tools/tree/master/msame)

# 推理
## 数据准备
    使用MNIST数据集进行训练和推理，数据集已上传于obs://sphere-face/MNIST_data/。
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
atc --model=/home/HwHiAiUser/sphereface/sphereface.pb \   #pb模型路径
    --framework=3 \
    --output=/home/HwHiAiUser/sphereface/sphereface \     #转换后OM模型输出路径
    --soc_version=Ascend310 \
    --input_shape="input_x:-1,28,28,1" \
    --log=error \
    --out_nodes="output:0" \
    --dynamic_batch_size="1,4,8,16,32" \
    --precision_mode=force_fp16
```

## msame推理命令
```commandline
#参照命令和实际环境设置路径和参数
msame --model=/home/HwHiAiUser/sphereface/sphereface.om \     #用于推理的OM模型路径
      --input=/home/HwHiAiUser/sphereface/bin/image_1 \            #用于推理的bin格式数据路径
      --output=/home/HwHiAiUser/msame/out/ \                    #推理结果输出路径
      --dymBatch=1                                              #batch大小
      --outfmt=TXT
```

## 精度计算
```commandline
python cal_inference_pref.py --LABEL_FLODER=XXX \   #label文件目录
                             --PREDICT_FLODER=XXX   #推理结果路径
```

# 附录
## 推理文件OBS路径：
   - pb模型: obs://sphere-face/SphereFace_ID0771_for_ACL/Appendix/pb/
   - om模型: obs://sphere-face/SphereFace_ID0771_for_ACL/Appendix/om/