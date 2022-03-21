## SRDRM模型离线推理

原始模型参考 [github链接](https://github.com/xahidbuffon/srdrm), 迁移训练代码到NPU之后进行离线推理
论文地址：https://ieeexplore.ieee.org/document/9197213

### Requirements

- CANN软件包版本：Ascend-cann-[xxx]_5.0.3.alpha005_linux-x86_64 
- Ascend 310
- 离线推理工具：[msame](https://gitee.com/ascend/tools/tree/master/msame)
- 其他依赖参考requirements.txt
- Dataset: USR-248(请参考https://github.com/xahidbuffon/srdrm自行下载)，以下实验均以测试数据集USR_8x为例
- Model: **SRDRM** and **SRDRM-GAN** for underwater image super-resolution 

### 1.准备原始模型文件

- `model.h5` (权重文件) 与 `model.json`(网络结构文件)
- 获取**srdrm** 在**npu**上训练的模型文件，包括h5,json文件。网盘链接中包含pb及om文件。[百度网盘，提取码：g3xp](https://pan.baidu.com/s/103CVyRhD4WB1cRPpNHSAvQ)
- 获取**srdrm-gan** 在**npu**上训练的模型文件，包括h5,json文件。网盘链接中包含pb及om文件。[百度网盘，提取码：iegf](https://pan.baidu.com/s/10BHahL19cmnmTVAvX4bxMQ)

### 2.代码路径解释

 ```shell
 |-- script
 |   |-- bin2img_measure.sh                   -- 处理推理结果bin文件并获得结果指标
 |   |-- freeze_graph.sh                      -- checkpoint转pb
 |   |-- jpg2bin.sh                           -- 测试数据集预处理
 |-- utils
 |   |-- data_utils.py                        -- 数据预处理文件
 |   |-- ssim_psnr_utils.py                   -- 计算SSIM PSNR
 |   |-- uqim_utils.py                        -- 计算uqim
 |-- bin2img_measure.py                       -- 推理结果后处理脚本
 |-- evaluate.py                              -- 评估脚本
 |-- img_preprocess.py                        -- 图像预处理脚本
 |-- keras_to_tensorflow.py                   -- 模型固化脚本
 |-- reqirements.txt                          
 |-- README.md  
 ```

### 3.数据集组织方式

>  离线推理只使用测试数据集 USR_8x。

```shell
数据集组织
|--- dataset
|   |-- SRDRM
|       |-- USR248
|           |-- TEST
|           |-- train_val
```

### 4.模型固化 (checkpoint转pb)

```shell
sh freeze_graph.sh | tee freeze_graph.log # 详细的参数信息请在freeze_graph.sh查看, 输出pb文件
```

### 5. `ATC`模型转换（pb模型转om模型）

1. 请按照[`ATC`工具使用环境搭建](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0004.html)搭建运行环境。

2. 参考以下命令完成模型转换。

   ```shell
   # srdrm生成模型atc模型转换命令
   cd /home/HwHiAiUser/Ascend/ascend-toolkit/5.0.3.alpha005/x86_64-linux/atc/bin/
   atc --model=/home/HwHiAiUser/module/model_52.pb --framework=3 --output=/home/HwHiAiUser/module/out/tf_srdrm_52_310_bz2 --soc_version=Ascend310  --input_shape="input_1:2,60,80,3" --output_type="conv2d_57/Tanh:0:FP32"  --out_nodes="conv2d_57/Tanh:0" 
   
   # srdrm-gan模型atc模型转换命令
   cd /home/HwHiAiUser/Ascend/ascend-toolkit/5.0.3.alpha005/x86_64-linux/atc/bin/
   atc --model=/home/HwHiAiUser/module/srdrmgan_model_g.pb --framework=3 --output=/home/HwHiAiUser/module/out/srdrmgan_310_bz2 --soc_version=Ascend310  --input_shape="input_4:2,60,80,3" --output_type="conv2d_65/Tanh:0:FP32"  --out_nodes="conv2d_65/Tanh:0" 
   # 其中可通过修改--input_shape参数来更改batchsize
   ```
   

​        **输出模型的路径为output参数所给的路径，各个参数的含义请参考3.1中给的链接**，

### 6.预处理图片，将图像转换为bin文件

   ```shell
   sh jpg2bin.sh | tee jpg2bin.log # 详细的参数信息请在jpg2bin.sh查看
   ```
   数据与处理后的bin文件下载方式: [百度网盘,提取码:i7n1](https://pan.baidu.com/s/10HwwhBWg_Vl0GAwLim-Rlg)
### 7.离线推理

1. 请先参考https://gitee.com/ascend/tools/tree/master/msame，编译出msame推理工具

2. 在编译好的msame工具目录下执行以下命令。

   ```shell
   # srdrm生成模型离线推理
   ./msame --model /home/HwHiAiUser/module/out/tf_srdrm_52_310_bz2.om --input /home/HwHiAiUser/AscendProjects/infer/input --output /home/HwHiAiUser/AscendProjects/infer/output/srdrm/ --outfmt BIN --debug true 2>&1 |tee inference_srdrm.log
   
   # srdrm-gan模型离线推理
   ./msame --model /home/HwHiAiUser/module/out/srdrmgan_310_bz2.om --input /home/HwHiAiUser/AscendProjects/infer/input --output /home/HwHiAiUser/AscendProjects/infer/output/srdrm_gan --outfmt BIN --debug true 2>&1 |tee inference_srdrmgan.log
   ```
   
   各个参数的含义请参考 https://gitee.com/ascend/tools/tree/master/msame
   
   > 此处因为篇幅原因，此处仅贴出srdrm模型batchsize为1和2的推理截图，其他bz结果及推理日志请参考[百度网盘链接，提取码：a4d6](https://pan.baidu.com/s/1biqZZObXZ7RVOPXmZcQI7A)
   
   **srdrm离线推理性能（bz=1）**:

<img src="https://gitee.com/windclub/image_bed/raw/master/img/20211114224917.png" align="center">

<img src="https://gitee.com/windclub/image_bed/raw/master/img/20211114224952.png" align="center"/>

​       **srdrm离线推理性能（bz=2）**:

<img src="https://gitee.com/windclub/image_bed/raw/master/img/20211114225020.png" align="center">

<img src="https://gitee.com/windclub/image_bed/raw/master/img/20211114225038.png" align="center"/>

3.离线推理性能对比【单张图片的推理时间(ms)】

| batchsize |   1    |   2    |   4    |   8    |   16   |
| :-------: | :----: | :----: | :----: | :----: | :----: |
|   srdrm   | 126.95 | 104.78 | 98.91  | 97.76  | 99.34  |
| srdrm-gan | 165.51 | 154.15 | 142.99 | 143.76 | 141.56 |




### 8.离线推理结果评估


    sh bin2img_measure.sh | tee measure.log # 详细的参数信息请在bin2img_measure.sh查看


模型离线推理指标【bz几乎不影响离线推理精度，故此处只列出单个精度值】

测试数据集USR_8X  

<table>
    <tr>
       <td>metrics</td>
       <td colspan="2" align="center">PSNR</td>
       <td colspan="2" align="center">SSIM</td>
       <td colspan="2" align="center">UQIM</td>
    </tr>
    <tr>
      <td></td>
      <td>npu_train</td>
      <td>offline_infer</td>
      <td>npu_train</td>
      <td>offline_infer</td>
      <td>npu_train</td>
      <td>offline_infer</td>
    </tr>
    <tr>
      <td>srdrm</td>
      <td>23.6960</td>
      <td>23.6852</td>
      <td>0.6301</td>
      <td>0.6302</td>
      <td>2.2626</td>
      <td>2.2644</td>
    </tr>
    <tr>
      <td>srdrm-gan</td>
      <td>21.3991</td>
      <td>21.4710</td>
      <td>0.5746</td>
      <td>0.5743</td>
      <td>2.9544</td>
      <td>2.9386</td>
    </tr>
</table>

