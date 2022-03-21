###   **基本信息** 

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing** 

**版本（Version）：1.2**

**修改时间（Modified） ：2022.1.20**

**框架（Framework）：TensorFlow 2.4.0**

**模型格式（Model Format）：ckpt**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的BigbiGAN网络训练代码** 


###   **概述** 
- BigbiGAN模型是以最先进的BigGAN模型为基础，通过添加编码器和修改鉴别器将其扩展到表征学习。作者广泛评估了BigbiGAN模型的表示学习和生成图像的能力，在ImageNet数据集上证明了它在无监督表示学习和无条件图像生成方面都达到了最先进的水平。模型包括生成器、鉴别器和编码器：生成器G和鉴别器D的组成部分均F来自BigGAN模型，G中有3个残差块，F中有4个残差块；鉴别器D中的H和J部分是带有跳跃连接的６层MLP(units=64) 。
- 参考文献： Large Scale Adversarial Representation Learning
- 参考实现： https://github.com/LEGO999/BIgBiGAN
- 论文中提到的主要贡献有四点：
1. 在ImageNet数据集上，BigBiGAN模型达到了无监督表征学习的最新技术水平
2. 为BigBiGAN提出了一个更稳定的联合判别器
3. 对模型设计的选择进行了全面的实证分析和消融实验
4. 表明了表征学习目标还有助于无条件图像生成，并展示了在ImageNet数据集上无条件图像生成的最新结果
- 网络架构：：
<img src="https://images.gitee.com/uploads/images/2021/1120/163144_f56100d0_8123771.png" style="zoom:30%;" />


### 默认配置<a name="section91661242121611"></a>
- 训练数据集预处理（以MNIST训练集为例，仅作为用户参考示例）：

  - 图像的输入尺寸为64*64

- 训练超参

  - Batch size： 256
  - Train epoch: 50

###   **训练环境准备** 
- Python 3.8.5  
- TensorFlow 2.4.0  
- tensorflow_datasets 2.1.0  
- Matplotlib 3.4.2  
- Numpy 1.19.5  
- absl-py 0.12.0  


### **数据集准备** 
训练数据集为MNIST数据集

### **脚本参数**
```
--dataset_path             //存放数据集的路径
--result_path              //存放结果的路径
--num_epochs               //训练的epoch数量
--train_batch_size         //用于训练的batch size大小
--num_classes              //数据集中的类别数量
--gen_disc_ch              //生成器和鉴别器的第一层中的通道数量
--en_ch                    //编码器的第一层中的通道数量
--lr_gen_en                //生成器的学习率
--beta_1_gen_en            //生成器优化器中的Beta_1参数
--beta_2_gen_en            //生成器优化器中的Beta_2参数
--lr_disc                  //鉴别器的学习率
--beta_1_disc              //鉴别器优化器中的Beta_1参数
--beta_2_disc              //鉴别器优化器中的Beta_2参数
```

###  **启动训练和测试过程**
测试示例：
```
python main.py --dataset_path tensorflow_datasets --result_path results --train_batch_size 256 --num_epochs 50
```
或
```
bash test/train_performance_1p.sh --dataset_path=tensorflow_datasets --result_path=results --train_batch_size=256 --num_epochs=50
```