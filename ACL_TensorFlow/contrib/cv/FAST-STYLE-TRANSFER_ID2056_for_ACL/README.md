<h2>Fast Style Transfer in TensorFlow</h2>

<h3>基本信息</h3>

发布者 : 华为 

应用领域 ：图像风格迁移

精度 ：Mixed

版本 ：1.0

模型格式 ：ckpt/pb/om

框架 ： TensorFlow(1.15.0)

处理器 ： 昇腾910

描述：基于TensorFlow框架的Fast Style Transfer图像风格迁移训练代码

<h3>简述</h3>                      

Fast-Style-Transfer模型是基于Ulyanov等人(2016)中介绍的快速风格化方法的改进。主要改进点在于将原始风格化架构中的Batch normalization改为Instance Normalization从而导致生成的图像质量得到了显著的改善。

* 参考论文：Ulyanov D, Vedaldi A, Lempitsky V. Instance normalization: The missing ingredient for fast stylization[J]. arXiv preprint arXiv:1607.08022, 2016.
* 参考实现：https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/ACL_TensorFlow/contrib/cv/FAST-STYLE-TRANSFER_ID2056_for_ACL

<h3>默认配置</h3>

* 训练超参
  * Batch size：20
  * Content-weight： 1.5e1
  * Checkpoint-iterations： 1000
  * Epoch： 2
  * Content_weight：7.5e0
  * Style_weight：1e2
  * TV_weight：2e2
  * Learning_rate：1e-3

<h3>示例代码</h3>

```
|——checkpoint                            //保存训练模型
|——examples
	|——content                           //示例内容图片
	|——results                           //迁移结果图片
	|——style                             //示例风格图片
	|——thumbs                            //示例图片
|——src
	|——optimize.py                       //模型训练模块
	|——transform.py					   //图片风格迁移模块
	|——utils.py                          //模型中使用到的一些基本函数
	|——vgg.py                            //损失网络VGG19模型
|——README.md						   //代码说明文档
|——boot_modelarts.py                      //模型启动文件
|——cfg.py                                //模型配置参数
|——evaluate.py                           //结果评估
|——help_modelarts.py                     //Obs通信模块
|——run_style.sh                          //训练脚本
|——style.py                              //训练文件
|——change_pb.py                          //ckpt转pb
|——tobin.py                              //数据转bin
|——tojpg.py                              //将最终结果bin转为jpg
```


<h3>Requirements</h3>

* TensorFlow 1.15.0
* Python 3.6
* Pillow 3.4.2
* scipy 0.18.1
* moviepy 1.0.2

<h3>模型训练与评估</h3>

模型需要与OBS之间进行数据传输，数据集和训练好的VGG模型都存在OBS上，运行时需传入ModelArts。

具体训练步骤如下：

ModelArts根据入口文件启动训练脚本
* 模型精度训练脚本

```
· modelarts_entry_acc.py 
os.system("dos2unix ./test/*")
os.system("bash ./test/train_full_1p.sh --data_path=%s --output_path=%s " % (config.data_url, config.train_url))
```

启动训练脚本:

```python
python modelarts_entry_acc.py
```

模型训练脚本如下：

```
·train_full_1p.sh
python3.7 ./style.py \                           #模型训练文件
        --style=${data_path}/wave.jpg \               #风格图片输入
        --checkpoint-dir=${output_path} \                   #保存训练模型
        --test=${data_path}/chicago.jpg \                   #测试图片输入
        --test-dir=${output_path} \                         #测试图片输出保存
        --content-weight 1.5e1 \                           #内容在loss函数中的权重
        --checkpoint-iterations 1000 \                     #迭代轮次
        --batch-size 20 \                                  #批量大小
        --train-path=${data_path}/train2014 \               #训练图片路径(完整数据集)
        --vgg-path=${data_path}/imagenet-vgg-verydeep-19.mat       #预训练VGG19模型路径
```

* 模型性能训练脚本

```
· modelarts_entry_perf.py
os.system("dos2unix ./test/*")
os.system("bash ./test/train_performance_1p.sh --data_path=%s --output_path=%s " % (config.data_url, config.train_url))
```

启动训练脚本:

```python
python modelarts_entry_perf.py
```

模型训练脚本如下：

```
·train_performance_1p.sh
python3.7 ./style.py \                           #模型训练文件
        --style=${data_path}/wave.jpg \               #风格图片输入
        --checkpoint-dir=${output_path} \                   #保存训练模型
        --test=${data_path}/chicago.jpg \                   #测试图片输入
        --test-dir=${output_path} \                         #测试图片输出保存
        --content-weight 1.5e1 \                           #内容在loss函数中的权重
        --checkpoint-iterations 1000 \                     #迭代轮次
        --batch-size 5 \                                  #批量大小
        --train-path=${data_path}/train_min_2014 \               #训练图片路径(简化数据集)
        --vgg-path=${data_path}/imagenet-vgg-verydeep-19.mat       #预训练VGG19模型路径
```




<h3>精度与性能</h3>

* 在GPU上运行大概4~5个小时，NPU运行40几小时
* 模型无特定的精度评价指标，主要通过观察来判断结果好坏                                                 

**风格**

<img src="./examples/style/wave.jpg" alt="wave" width="500" height="313"/>

**原图**

<img src="./examples/content/chicago.jpg" alt="chicago" width="500" height="313"/>

**GPU结果**

<img src="./examples/output/chicago_gpu.jpg" alt="chicago_gpu" width="500" height="313" />

**NPU结果**

<img src="./examples/output/chicago_npu.png" alt="chicago_npu" width="500" height="313" />




<h3>推理</h3>

* 首先根据模型ckpt的存储目录，运行change_pb.py文件，将ckpt文件转为pb文件，冻结模型参数

  ```python
  python change_pb.py --ckpt_path ./checkpoint/fns.ckpt
  ```

  

* 数据预处理，运行tobin.py文件将.jpg文件转为.bin文件

  ```python
  python tobin.py 
  ```

  

* 在华为云镜像服务器上将pb文件转为om文件

* 应用msame工具运行模型推理

  ```python
  ./msame --model ./path/to/om --input ./path/to/input/bin --output ./path/to/output/bin --outfmt BIN --loop 1
  ```

  

* 得到最终结果，文件格式为bin，运行tojpg.py将bin转为jpg，即可查看最终图像效果

<h3>推理结果</h3>
  
**GPU**

<img src="./examples/output/GPU.jpg" alt="GPU"/>

**推理**

<img src="./examples/output/inference.jpg" alt="npu"/>


<h3>推理性能</h3>

<img src="./examples/output/yunx.jpg" alt="yx"/>
