# 环境
推理前请参考下文进行环境配置: [使用ATC转换OM模型](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0004.html) and [使用msame进行推理](https://gitee.com/ascend/tools/tree/master/msame)

# 推理
## 数据准备
    几张bmp格式图片
## 使用data_convert_to_bin.py进行数据预处理
```commandline
#图像经过预处理后分辨率储存在文件名中
#test_dir： 数据集路径
#save_dir： bin文件储存路径
python data_convert_bin.py --test_dir=XXX --ave_dir=XXX
```
## OM模型转换命令
```commandline
#参照命令和实际环境设置具体路径和参数
atc --model=/home/HwHiAiUser/code/pb/RetinexNet.pb \   #pb模型路径
    --framework=3 \
    --output=/home/HwHiAiUser/code/pb/RetinexNet \     #转换后OM模型输出路径
    --soc_version=Ascend310 \
    --input_shape="input_low:1,680,720,3" \
    --log=info \
    --out_nodes="mul:0"
```

## msame推理命令
```commandline
#参照命令和实际环境设置路径和参数
msame --masme_path=/home/HwHiAiUser/msame/tools/msame/out \
      --model=/home/HwHiAiUser/cowmask/RetinexNet.om \     #用于推理的OM模型路径
      --input=/home/HwHiAiUser/cowmask/bin/image_1 \            #用于推理的bin格式数据路径
      --output=/home/HwHiAiUser/msame/out/ \                    #推理结果输出路径
      --dymBatch=1                                              #batch大小
```

## 图像复原
```commandline
high_light_figure.py --atc_dir=XXX \   #推理结果目录
                     --width=XXX   #图像宽度
                     --height=XXX   #图像高度
```
# 附录
## 推理文件OBS路径：
   - pb模型: obs://retinexnet/ACL/RetinexNet.pb
   - om模型: obs://retinexnet/ACL/RetinexNet.om

链接：https://pan.baidu.com/s/1ylTcivi1XM-1UIu-yRwDjA 
提取码：rtxn