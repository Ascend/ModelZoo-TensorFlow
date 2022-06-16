## Pix2Vox

### 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：**3D Object Reconstruction

**版本（Version）：1.0**

**修改时间（Modified） ：2022.6.16**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：om**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的三维物体重建**

### 概述

A Novel Hybrid Ensemble Approach For 3D Object Reconstruction from Multi-View Monocular RGB images for Robotic Simulations. 

- 参考论文：

  https://arxiv.org/pdf/1901.11153.pdf

- 参考实现：

  https://github.com/Ajithbalakrishnan/3D-Object-Reconstruction-from-Multi-View-Monocular-RGB-images

### 文件结构
  ```
  |- combinefloat32.bin                      # 离线推理数据数据，由三张图片拼接转化成
  |- my_tools.py                             # 本地工具集
  |- pb.py                                   # ckpt模型转pb模型代码
  |- README                                  # 项目说明
  |- tobin.py                                # 数据转化程序
  |- om_model_output_0.txt                   # 离线推理案例结果
  ```

### ckpt转pb
将ckpt文件转为pb文件，运行代码：
```
Python pb.py
```
### pb转om
将pb模型转为om模型:
```
atc --model=/home/HwHiAiUser/AscendProjects/pix2vox/pb_model/frozen_model.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/pix2vox/om_model --soc_version=Ascend310 --input_shape="Placeholder:1,3,127,127,3" --log=info --out_nodes="ref_net/ref_Dec/ref_out:0"
```

### msame工具
我们采用msame工具进行离线推理，参考[msame简介](https://gitee.com/ascend/tools/tree/master/msame), 获取msame推理工具及使用方法。

由于版本原因，使用的是老版的msame工具，将msame工具zip压缩包上传到ecs离线服务器的/root/目录下，执行以下命令：

```
unzip msame.zip
env |grep DDK_PATH
env |grep NPU_HOST_LIB
mv msame/ AscendProjects
cd $HOME/AscendProjects/msame/
chmod +x *
./build.sh g++ $HOME/AscendProjects/msame/out
```

### 数据集转bin
网络输入的形状是[1,3,127,127,3]，即将三张图片读取为numpy数组格式，数据类型为float32，将三张图片拼接到一起，最后再reshape，数据集转换运行代码：
```
Python tobin.py
```
在离线推理服务器上/root/目录下新建datatest文件夹，将转化好的bin文件上传到该文件夹中。
### 推理测试
运行命令前在root目录下新建model_output文件夹作为网络输出目录，使用msame推理工具，参考如下命令，发起推理测试：
```
$HOME/AscendProjects/msame/out/msame  --model "/root/om_model.om"  --input "/root/datatest/combinefloat32.bin"  --output "/root/model_output/" --outfmt TXT --loop 1
```
运行完命令后网络输出结果储存在om_model_output_0.txt文件中，保存了一个大小为32×32×32矩阵，矩阵中元素的位置和大小就表示三维重建结果，数值（已归一化）越大表示此处存在目标的概率越大。

推理测试成功结果如下：

![image-20220616171552469](C:\Users\57239\AppData\Roaming\Typora\typora-user-images\image-20220616171552469.png)

# 三维重建结果展示

展示的案例输入的三张图片如下所示：

![00](https://gitee.com/zhangwx21/ModelZoo-TensorFlow/blob/master/ACL_TensorFlow/contrib/cv/Pix2Vox_ID1284_for_ACL/image/00.png)![01](E:\Desktop\ShapeNetRendering\1111\test\rendering\01.png)![02](E:\Desktop\ShapeNetRendering\1111\test\rendering\02.png)

神经网络模型三维重建结果如图所示：

![image-20220616203111652](C:\Users\57239\AppData\Roaming\Typora\typora-user-images\image-20220616203111652.png)



备注：

本项目源代码参考实现并非基于论文源代码（只提供了pytorch版本），参考的源代码也没有给出推理程序和可视化程序，因此无法进行推理结果精度对比。
