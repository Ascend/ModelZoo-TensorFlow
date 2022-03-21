
<h2 id="概述.md">概述</h2>

Multi-level Spatially-Pooled (MLSP) features extracted from ImageNet pre-trained Inception-type networks are used to train aesthetics score (MOS) predictors on the Aesthetic Visual Analysis (AVA) database. The code shows how to train models based on both narrow and wide MLSP features.

- This is part of the code for the paper "Effective Aesthetics Prediction with Multi-level Spatially Pooled Features". Please cite the following paper if you use the code:
    
    ```
    @inproceedings{hosu2019effective,
  title={Effective Aesthetics Prediction with Multi-level Spatially Pooled Features},
  author={Hosu, Vlad and Goldlucke, Bastian and Saupe, Dietmar},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9375--9383},
  year={2019}}
    ```

# om模型转换
在Ascend310推理服务器下，使用ATC模型转换工具进行模型转换:
```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1

atc --model=/home/HwHiAiUser/AscendProjects/final_model.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/final_model --soc_version=Ascend310 --input_shape="input_1:1,514, 800, 3"
```
请从此处下载[pb模型](https://myd-ava.obs.cn-north-4.myhuaweicloud.com/model/final_model.pb)

请从此处下载[om模型](https://myd-ava.obs.cn-north-4.myhuaweicloud.com/model/final_model1.om)


# 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。


## 1.数据集转换bin
原始数据集[链接](https://github.com/imfing/ava_downloader)

下载好原始数据集后，选取metadata文件下AVA_data_official_test.csv标注为test的图片，选取并
将其保存在dataset目录下，并执行`toBin.py`文件将需要测试的图片转换为推理需要的——>bin数据


## 2.推理

使用msame推理工具，发起推理测试，推理命令如下：

```bash
cd /home/HwHiAiUser/AscendProjects/tools/msame/out
./msame --model "/home/HwHiAiUser/AscendProjects/final_model1.om" --input "/home/HwHiAiUser/AscendProjects/bindata/0.bin" --output "/home/msame/out/" --outfmt TXT
```

## 3.推理结果

```
root@yuanma:/home/msame/out# ./msame --model "/home/HwHiAiUser/AscendProjects/final_model1.om" --input "/home/HwHiAiUser/AscendProjects/bindata/0.bin" --output "/home/msame/out/" --outfmt TXT
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model /home/HwHiAiUser/AscendProjects/final_model1.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
/home/msame/out//20211222_104250
[INFO] start to process file:/home/HwHiAiUser/AscendProjects/bindata/0.bin
[INFO] model execute success
Inference time: 504.977ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 504.977000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl


```

## 4。查看测试图片质量mos分数
打开推理生成的txt文件，即可以得到om模型所输出对于该图片的测评分数。
例如：本次测评的954013.jpg，经过推理之后的最终分数为：6.32812  