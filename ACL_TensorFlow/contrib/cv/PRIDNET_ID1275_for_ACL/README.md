## PRIDNET模型离线推理

PRIDNET网络是在论文《Pyramid Real Image Denoising Network》中提出的。论文进行了系统的研究，解决了深度卷积神经网络 (CNN)在现实世界的噪点图像处理不佳的问题。PRIDNet的主要贡献包括以下三方面：(1)Channel Attention：在提取的噪声特征上利用通道注意机制，自适应地校正通道重要性，解决大多数基于CNN的去噪方法中所有通道的特征被平等对待的问题。(2)Multi-scale feature extraction：设计了一种金字塔去噪结构，其中每个分支都关注一个尺度的特征。利用它可以同时提取全局信息并保留局部细节，从而为后续的全面去噪做好准备。多尺寸特征提取解决了固定感受野无法携带多样信息的问题。(3)Feature self-adaptive fusion：级联的多尺度特征，每个通道代表一个尺度的特征，引入了核选择模块。采用线性组合的方法对不同卷积核大小的多分支进行融合，使得不同尺度的特征图可以通过不同的核来表达。特征自适应融合，解决了大多数方法对不同尺寸的特征进行不加区分的处理，不能自适应的表达多尺度特征的问题。

### 推理环境

- CANN软件包版本：Ascend-cann-[xxx]_5.0.4.alpha001_linux-x86_64 
- Ascend 310
- atc转模工具：请参考：[ATC快速入门_昇腾CANN社区版(5.0.3.alpha002)(推理)_ATC模型转换_华为云](https://support.huaweicloud.com/atctool-cann503alpha2infer/atlasatc_16_0005.html)
- 离线推理工具：[msame](https://gitee.com/ascend/tools/tree/master/msame)
- 数据集：SIDD-Medium Dataset  Raw-RGB images only (~20 GB)，下载地址为：[https://www.eecs.yorku.ca/~kamel/sidd/dataset.php](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)

### 代码及路径解释

 ```shell                    
 |-- freeze_graph.py                       -- 模型固化代码(checkpoint转pb文件)
 |-- data_convert2bin.py                   -- 数据预处理代码(生成bin文件)
 |-- run_act.sh                            -- atc转模脚本(pb文件转om模型)
 |-- run_msame.sh                          -- msame离线推理脚本
 |-- store2mat.py                          -- 将txt文件转为mat文件代码
 |-- test.py                               -- 计算SSIM和PSNR代码
 |-- reqirements.txt                          
 |-- README.md  
 ```

### 模型固化 (checkpoint转pb文件)

```shell
python3.7 freeze_graph.py
```
PRIDNET在npu上训练的checkpoint文件及固化pb文件地址为：链接：obs://pridnetdata/ACL/checkpoint文件及固化pb文件/


### atc模型转换(pb文件转om模型)

1. 执行shell脚本将pb文件转换为om模型

   ```shell
   sh run_act.sh
   ```
2. shell脚本中的atc命令参数请参考：[参数说明](https://support.huaweicloud.com/atctool-cann503alpha2infer/atlasatc_16_0038.html)

### 将输入数据转换为bin文件

 运行data_convert2bin.py生成bin文件

   ```shell
   python3.7 data_convert2bin.py
   ```

   数据预处理后生成的bin文件地址为: 链接：obs://pridnetdata/ACL/bin文件/data/

### 离线推理

1. 请参考https://gitee.com/ascend/tools/tree/master/msame，安装msame推理环境

2. 编译成功之后，将run_msame.sh上传至msame工具的out目录下执行

   ```shell
   sh run_msame.sh
   ```
   
   shell脚本中的msame命令参数请参考:https://gitee.com/ascend/tools/tree/master/msame


   推理性能：38.592ms

3. 运行store2mat.py将txt文件转为mat文件

   ```shell
   python3.7 store2mat.py
   ```
4. 运行test.py计算SSIM和PSNR

   ```shell
   python3.7 test.py
   ```
### 推理精度
<table>
    <tr>
       <td>metrics</td>
       <td colspan="2" align="center">PSNR</td>
       <td colspan="2" align="center">SSIM</td>
    </tr>
    <tr>
      <td>chip</td>
      <td>npu</td>
      <td>推理</td>
      <td>npu</td>
      <td>推理</td>
    </tr>
    <tr>
      <td>PRIDNET</td>
      <td>45.0205</td>
      <td>45.0201</td>
      <td>0.9985</td>
      <td>0.9985</td>
    </tr>
</table>

