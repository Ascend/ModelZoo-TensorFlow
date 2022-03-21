## BSRN模型离线推理

原始模型参考 [github链接](https://github.com/idearibosome/tf-bsrn-sr/), 迁移训练代码到NPU之后进行离线推理
                                                                                                                                                                                                                                                                

### Requirements

- CANN软件包版本：Ascend-cann-5.0.4.alpha005_linux-x86_64 
- Ascend 310
- 离线推理工具：[msame](https://gitee.com/ascend/tools/tree/master/msame)
- 其他依赖参考requirements.txt
- Dataset: 推理数据集：BSD100，推理时用的就是此数据集
- Model: BSRN
                                                                                                                                                                                                                                                                

### 1.准备原始模型文件

- 获取BSRN模型checkpoint文件，该模型是在V100服务器上训练得到的
- 下载链接如下：[OBS链接，提取码：000000](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=8yJ8F9PuJGLkXZyo6CFso5eG39bP9z/TMx2alm87JBXBxvm5m4TO0lVLsXztbrj7WRMfuP+pgFRTWoa75Ruroa3w5RVx2IYP1bgIXgnuE4jhPND2Ruv0QqgwOVugW/qurIPe6Aw1VQdt/DFVkS4zlfyXbgTXPeeVOgRI/c7DedjKZ7xnQCEm4994oVP3jRvCEjHKIX50fAAJdAuSnCqsqul0fuMyLvsfcMJxXAwwd7zePqHD4/1jp65MSa6PFrq7n+p7EtKpARWzm+M4P+G+0k5q3n+KVlR+SW8kbTLGn75BT2g0hcUnHOWNg6B8P1Nr9mbOvl+hSSgv2e+YApQN5WSSo94XOoBzKKaabzyEgmp7fvbdFOod2VU9+gyEPkWFjixAn8Qk2SNg0cjdCHh4jOLUQ3mA89wOoDpYWZNH6ULP04DffuzzGcjg5I66Qnrnw9TwI0gIO2+j6v/tor6jdI1P3H0fOuvzUL5/7KMkSqVqY1WVbO+Zv5YK15b9DtlVjUKPruLoIIFcieC8uiTCs9mJGpq4Og3+euoUSSxhjkNEQUT7jlLkapr0oRDRUYoj)                                                                                                                                                                                                                            
### 2.代码路径解释

 ```shell
 |-- temp
 |   |-- results                  			 -- 存放模型checkpoint文件
 |   |-- dataset								 -- 存放推理数据集
 |-- tf-bsrn-sr
 |   |-- dataloaders                      	 -- 数据加载
 |   |-- models                   			 -- 模型文件
 |   |-- infer.py                      		 -- 计算指标脚本（推理结果后处理）
 |   |-- pb_validate_bsrn.py                 -- 固化模型验证
 |-- freeze.py                        		 -- 模型固化脚本
 |-- preprocess.py                   		 -- 图像预处理脚本
 |-- reqirements.txt                          
 |-- README.md  
 ```

### 3.数据集组织方式

>  离线推理使用测试数据集 BSD100

```shell
数据集组织
|-- dataset
|   |-- BSD100
|       |-- LR
|       |-- SR
```

### 4.模型固化 (checkpoint转pb)

```shell
python freeze.py
```

freeze脚本包含生成pb模型和测试两部分, 路径已提前硬编码到了脚本中，只需提前将checkpoint文件保存下来到temp/results目录下即可运行该脚本。通过该脚本可得到frozen_model.pb模型文件。若固化模型脚本运行正常，终端会输出''out:(160, 240, 240, 3)''，如下图所示：

<img src="https://raw.githubusercontent.com/coelien/image-hosting/master/img/202202281302971.png" alt="image-20220228130257939" style="zoom:50%;" align="center">

```
python pb_validate_bsrn.py
```

--dataloader=basic_loader
--data_input_path=temp/dataset/BSD100/LR
--data_truth_path=temp/dataset/BSD100/SR
--restore_path=temp/results/model.ckpt-1000000
--model=bsrn
--scales=4
--save_path=temp/results/result-pictures

目录需自行调整，参数设置可以参考如上进行，通过将pb文件进行读取并验证，可以判断生成的pb文件是否正确：

<img src="https://raw.githubusercontent.com/coelien/image-hosting/master/img/202202282032088.png" alt="image-20220228203218038" style="zoom:50%;" align="center">

### 5. `ATC`模型转换（pb模型转om模型）

1. 请按照[`ATC`工具使用环境搭建](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0004.html)搭建运行环境。
   
2. 参考以下命令完成模型转换。
                                                                                                                                                                                                                                                                
   ```shell
   # bsrn生成模型atc模型转换命令
   cd /home/HwHiAiUser/Ascend/ascend-toolkit/5.0.3.alpha005/x86_64-linux/atc/bin/
   atc --model=/root/HuaweiModelTransfer/freeze-models/frozen_model.pb --framework=3 --output=/root/HuaweiModelTransfer/freeze-models/bsrn-model-fixed --soc_version=Ascend310 --input-shape="sr_input:1,120,80,3"
   ```
   
   通过该命令可得到 bsrn-model-fixed.om文件
   
   实验结果图展示：
   
   <img src="https://raw.githubusercontent.com/coelien/image-hosting/master/img/202202281301209.png" alt="image-20220228130124075" style="zoom:50%;" align="center">                                                                                                                                            

### 6.预处理图片，将图像转换为bin文件

   ```shell
   python preprocess.py
   ```
   运行上面命令的必要参数如下：前两个代表数据集，后两个代表生成的bin文件目录

--data_input_dir=temp/dataset/BSD100/LR
--data_truth_dir=temp/dataset/BSD100/SR
--bin_input_dir=temp/bin/BSD100/LR
--bin_truth_dir=temp/bin/BSD100/SR

生成的bin文件目录如下：

```
├── BSD100
│   ├── LR
│   │   ├── x2
│   │   ├── x3
│   │   └── x4
│   └── SR
└── scale
```

### 7.离线推理

1. 请先参考https://gitee.com/ascend/tools/tree/master/msame，编译出msame推理工具
   
2. 在编译好的msame工具目录下执行以下命令。                                                                                                                                                                                                                                                       
   ```shell
   # bsrn生成模型离线推理
   ./msame --model /root/bsrn-model-fixed.om --output "/root/bsrn" --input "/root/bin/BSD100/LR/x4,/root/bin/scale" --debug true
   ```
   
   各个参数的含义请参考 https://gitee.com/ascend/tools/tree/master/msame
   
   **bsrn离线推理性能**:
   
   <img src="https://raw.githubusercontent.com/coelien/image-hosting/master/img/202202281905267.png" alt="image-20220228190555212" style="zoom:50%;" align="center">
   
   ​                                                                                                                                                                                                                   

<img src="https://raw.githubusercontent.com/coelien/image-hosting/master/img/202202281907943.png" alt="image-20220228190704896" style="zoom:50%;" align="center">


​                                                                                                                                                                                                                                                                

### 8.离线推理结果评估


    python infer.py #下面的参数路径需自行进行调整

<img src="https://raw.githubusercontent.com/coelien/image-hosting/master/img/202202281932965.png" alt="image-20220228193237922" style="zoom:50%;" align="center">

--dataloader=basic_loader
--data_input_path=temp/dataset/BSD100/LR
--data_truth_path=temp/dataset/BSD100/SR
--model=bsrn
--scales=4
--save_path=test_results
--output_path=2022222_12_53_32_755923/
--truth_path=temp/bin/BSD100/SR/

其中：truth_path是标签所生成的bin文件所在的目录;output_path是msame推理结果所在目录

下面是在测试数据集BSD100上进行离线推理和直接使用pb_validate进行推理的指标对比：                                                                                                                                                                                                                                                                

<table>
    <tr>
       <td>metrics</td>
       <td colspan="2" align="center">PSNR</td>
       <td colspan="2" align="center">SSIM</td>
    </tr>
    <tr>
      <td></td>
      <td>gpu_train</td>
      <td>offline_infer</td>
      <td>gpu_train</td>
      <td>offline_infer</td>
    </tr>
    <tr>
      <td>bsrn</td>
      <td>27.525</td>
      <td>26.444</td>
      <td>0.708</td>
      <td>0.680</td>
    </tr>
</table>
