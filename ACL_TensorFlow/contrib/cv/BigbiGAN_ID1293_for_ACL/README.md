## BigbiGAN模型离线推理
                                                                                                                                                                                                                                                                
### 概述
- BigbiGAN模型是以最先进的BigGAN模型为基础，通过添加编码器和修改鉴别器将其扩展到表征学习。作者广泛评估了BigbiGAN模型的表示学习和生成图像的能力，在ImageNet数据集上证明了它在无监督表示学习和无条件图像生成方面都达到了最先进的水平。模型包括生成器、鉴别器和编码器：生成器G和鉴别器D的组成部分均F来自BigGAN模型，G中有3个残差块，F中有4个残差块；鉴别器D中的H和J部分是带有跳跃连接的６层MLP(units=64) 。
- 参考文献： Large Scale Adversarial Representation Learning
- 参考实现： https://github.com/LEGO999/BIgBiGAN

### Requirements

- CUDA版本: 10.1
- tensorflow: 2.3.0
- Ascend 310
- 离线推理工具：[msame](https://gitee.com/ascend/tools/tree/master/msame)
- 其他依赖参考requirements.txt
- Model: BigbiGAN
                                                                                                                                                                                                                                                                
创建环境参考：
```
conda create -n tf2.5 python=3.8 -c conda-forge
conda activate tf2.5
pip install tensorflow-gpu==2.3.0
conda install cudatoolkit=10.1 cudnn=7.6 -c conda-forge
```
                                                                                                                                                                                                                          
### 1.代码路径解释

 ```shell
 |-- models                               --模型文件
 |-- pb_validate.py                       -- 固化模型验证
 |-- pic2bin.py                           -- 将输入转为.bin文件
 |-- getAcc.py                            -- 精度验证脚本（推理结果后处理）
 |-- reqirements.txt                          
 |-- README.md  
 ```

### 2.将输入转为bin文件

-  BigbiGAN模型包含若干个小模型:生成器、鉴别器和编码器。本次离线推理使用生成器完成图像生成的任务。
-  生成器的输入和输出：
 ```
    输入：随机噪声（fake_images：shape[256,100]） + 标签（label：shape[256]）
    输出：生成图片（shape：[256,32,32,1]）
 ```
-  运行pic2bin.py文件将输入的随机噪声fake_images和标签label转为.bin文件
```shell
  python pic2bin.py --output_bin_dir /home/ma-user/work/Bin
```
参数解释：
```
--output_bin_dir   //bin输出文件夹，默认在该目录下生成fake_image.bin和label.bin
```

### 3.模型固化（转pb）

-  执行BigbiGAN模型训练，完成后train.py文件中的save_pb函数将生成器保存为pb。
-  同时采用步骤2生成的bin文件，输入到训练完毕的生成器中，保存生成图像作为om精度对比的标准。
```shell
  python models/main.py --input_bin_dir /home/ma-user/work/Bin --dataset_path /home/ma-user/work/models/tensorflow_datasets --result_path /home/ma-user/work/models/results --train_batch_size 256 --num_epochs 50 --output_pb_path /home/ma-user/work/models/pb --output_gpu_generated_img /home/ma-user/work/models/generated_images 
```
参数解释：
```python
--input_bin_dir            //输入bin的目录路径
--dataset_path             //存放数据集的路径
--num_epochs               //训练的epoch数量
--train_batch_size         //用于训练的batch size大小
--result_path              //存放结果的路径
--output_pb_path           //存放输出pb的目录路径，默认在该目录下保存为bigbigan.pb
--output_gpu_generated_img //存放gpu训练生成图片的路径，默认在该目录下保存为gpu_out.bin
```
通过读取pb文件，可以输出pb生成的图片，在getAcc.py中与gpu标准对比，验证是否正确：
```shell
  python pb_validate.py --input_bin_dir /home/ma-user/work/Bin --input_pb_path /home/ma-user/work/models/pb --output_pb_generated_img /home/ma-user/work/models/generated_images 
```
参数解释：
```python
--input_bin_dir            //输入bin的目录路径
--input_pb_path            //输入pb的目录路径
--output_pb_generated_img  //存放pb生成图片的目录路径，默认在该目录下保存为pb_out.bin
```

### 4. `ATC`模型转换（pb模型转om模型）

1. 请按照[`ATC`工具使用环境搭建](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0004.html)搭建运行环境。
   
2. 参考以下命令完成模型转换。
                                                                                                                                                                                                                                                                
   ```shell
       atc --model=bigbigan.pb --framework=3 --output=./out/bigbigan --soc_version=Ascend310 --input_shape="x1:256,100;x2:256"
   ```
   
   通过该命令可得到 bigbigan.om文件
   
   实验结果图展示：
   ![输入图片说明](pic/atc.png)                                                                                                                                     


### 5.离线推理

1. 请先参考https://gitee.com/ascend/tools/tree/master/msame，编译出msame推理工具
   
2. 在编译好的msame工具目录下执行以下命令。                                                                                                                                                                                                                                                       
   ```shell
   cd $HOME/AscendProjects/tools/msame/
   ./msame --model "/home/bigbigan/out/bigbigan.om" --input "/home/bigbigan/bin/fake_image.bin,/home/bigbigan/bin/label.bin" --output "/home/bigbigan/msame/" --outfmt BIN --loop 1
   ```
   
   各个参数的含义请参考 https://gitee.com/ascend/tools/tree/master/msame
                                                                                                                                                                                                                                                                

### 6.离线推理精度性能评估

-  运行getAcc.py文件得到推理生成的图片，以及GPU、pb和om生成图片的精度对比
```shell
  python getAcc.py --input_om_out /home/ma-user/work/om_out/bigbigan_output_0.bin --input_gpu_generated_img /home/ma-user/work/models/generated_images/gpu_out.bin --input_pb_generated_img /home/ma-user/work/models/generated_images/pb_out.bin --output_om_generated_img /home/ma-user/work/om_out/generated_images
```
参数解释：
```shell
--input_om_out             /输入om推理输出的目录路径
--input_gpu_generated_img  //输入gpu生成图片的目录路径
--input_pb_generated_img   //输入pb生成图片的目录路径
--output_om_generated_img  //存放离线推理生成图片的目录路径
```
**离线推理精度**：

经GPU训练后生成的图像与pb，om模型的生成图像相比，平均像素值的差值均在10-4数量级，生成图片相似度高，误差极小。

![输入图片说明](pic/precision.png)

**离线推理性能**：

![输入图片说明](pic/inference_perf.png) 

**生成图像（前8张）**：从左至右-GPU、pb、om 

![输入图片说明](pic/generated_image.png)


### 推理模型下载

训练30个epoch的结果输出： 

百度网盘链接: https://pan.baidu.com/s/1MYcdjz-imfGaXeBjAqurKw 提取码: a948 