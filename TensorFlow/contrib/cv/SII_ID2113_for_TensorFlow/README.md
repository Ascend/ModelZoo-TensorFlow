# SII_IDhw52357014_for_tensorflow

# 基本信息

**发布者(publisher):Huawei**

**应用领域（Application Domain）： Image Inpaint**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.12.19**

**大小（Size）：106k**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt/om(msame支持的模型格式)**

**处理器（Processor）：昇腾310**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的DCGAN图像修复代码**

## 概述

- 参考论文：

  [Semantic Image Inpainting with Deep Generative Models, CVPR2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.pdf).

- ```
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 默认配置

- 训练数据集预处理（以SVHN为例，仅作为用户参考示例）：
  - 图像的输入尺寸为64*64
  - 图像输入格式：mat（Matlab保存数据的格式）
- 测试数据集预处理（以SVHN为例，仅作为用户参考示例）
  - 图像的输入尺寸为64*64
  - 图像输入格式：mat（Matlab保存数据的格式）
- 训练超参
  - Batch size: 256
  - Learning Rate:2e-4
  - Momentum:0.5
  - Iters:2000000

## 训练环境准备

```shell
conda create -n SII python=3.7
conda activate SII
pip install tensorflow-gpu==1.15.0 opencv-python scikit-image matplotlib pyamg scipy
```

## 快速上手

1. 下载源文件
2. 下载数据集
   请自行下载数据集 svhn，并且按照下面的格式放好数据集
   Two mat files you need to download are `train_32x32.mat` and `test_32x32.mat` in Cropped Digits Format 2.

## 模型训练

1. 使用训练作业的形式去训练
   使用Ascend-Powered-Engine | tensorflow_1.15-cann_5.0.2-py_ 的AI引擎，然后数据集的位置代码目录的格式应该是

   ```
   .
   │   semantic_npu
   │   ├── src
   │   │   ├── dataset.py
   │   │   ├── dcgan.py
   │   │   ├── download.py
   │   │   ├── inpaint_main.py
   │   │   ├── inpaint_model.py
   │   │   ├── inpaint_solver.py
   │   │   ├── main.py
   │   │   ├── solver.py
   │   │   ├── mask_generator.py
   │   │   ├── poissonblending.py
   │   │   ├── tensorflow_utils.py
   │   │   └── utils.py
   │   |   Data
   │   |   ├── celebA
   │   |   │   ├── train
   │   |   │   └── val
   │   |   ├── svhn
   │   |   │   ├── test_32x32.mat
   │   |   │   └── train_32x32.mat
   ```

   本代码测试的是svhn的数据集
   然后直接将semantic_npu作为代码目录，然后启动文件为main.py，然后没有输入与输出目录

## 训练模型

model.pb
链接：https://pan.baidu.com/s/13EAPoq0KWQQivQCgaKwz0g 
提取码：1234

ckpt模型
链接：https://pan.baidu.com/s/1oSfBZ1fNRT36pnxQgNCmcA 
提取码：1234

om文件
链接：https://pan.baidu.com/s/1mi4rcsUb-hjeMHNrrWFy8Q 
提取码：1234

# 注意事项

```python
def main(_):
    print(os.getcwd())
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index
    model_dir = "/svhn/"
    src=current_path+model_dir
    print(src)
    print(os.path.isdir(src))
    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
        model_dir = "/cache/svhn/"
        # 训练结束后，将ModelArts容器内的训练输出拷贝到OBS
        cur_time = datetime.now().strftime("%Y%m%d-%H%M")
        mox.file.copy_parallel(model_dir, "obs://liu-ji-hong-teacher/"+cur_time+"/svhn")
#         mox.file.copy_parallel(current_path+"/kernel_meta/", "obs://liu-ji-hong-teacher/semantic-finish/kernel_meta")
    else:
        solver.test()
```

上面代码的obs链接需要换为自己的，这条命令就是将训练好的模型以及生成的图像的文件夹拷贝到obs桶里

# 使用模型生成inpaint的图像

python3.7 inpaint_main.py --load_model=20211017-2152 --mask_type=center
这个里面的load_model是在生成的src/svhn的存放模型的文件夹
本文论文中使用的四种mask所以需要训练一次修改一次mask

```python
#inpaint_main.py
tf.flags.DEFINE_string('mask_type', 'center', 'mask type choice in [center|random|half|pattern], default: center')
```

需要在每次训练作业时修改mask_type,这样就可以生成四种mask的图片，要把生成的图片分开保存，用来后面测试的时候使用
这样一会会生成四组mask的

# 测试模型

```shell
python3 SSIM2.py ${mask_type}
```

SSIM2.py为自己写的求图片的PSNR指标的文件，而后面的random指的是需要测试的mask

在调用inpaint_main.py时会产生许多npy包含的生成的图像以及inpaint后 的图像，从而SSIM2.py的作用就是读取npy然后求图像的PSNR指标的值

# 离线推理

1. 把生成的模型转换为可以用于离线推理的模型

   `python3 test_main.py --load_model=${model_path} --is_test=True`

   * model_path指的是在训练完成后会在**src/svhn/model/**里面生成一个以训练开始时的日期为名字的文件夹，假设为20211215-0005，则`export model_path=20211215-0005`

2. atc转换模型的命令：

`atc --model=model.pb --framework=3 --output=model/semantic --soc_version=Ascend310  --input_shape="latent_vector:2,100;wmasks:2,64,64,3;images:2,64,64,3" --out_nodes="out_3_context_loss:0;out_4_prior_loss:0;g_/out_1_tanh:0"`

3. 离线推理的命令

   先在.bashrc或者/etc/profile文件后添加

   ```shell
   export ASCEND_SLOG_PRINT_TO_STDOUT=0
   export ASCEND_GLOBAL_LOG_LEVEL=3
   #然后
   source ~/.bashrc
   #这一步是因为如果不设置，会在~/ascend生成log文件夹，由于推理会需要msame推理1500*20生成的log会很大，最后会报错
   ```

   

   `python3 test_main.py --load_model=${model_path}`

   > 之所以会在离线推理时还需要导入tensorflow的ckpt的模型，是因为 推理部分只做正向推理部分，然后套壳用cpu计算反向梯度 + 310推理 完成最终目标。 因为310 不支持反向计算。

# 训练性能对比（s/iter)

注：batch为256

| paltfrom | 性能（一次迭代需要的时间） | 平均每次迭代所需的时间 |
| -------- | -------------------------- | ---------------------- |
| gpu      | 0.58-0.70                  | 0.60                   |
| npu      | 0.54-0.68                  | 0.59                   |

# 训练精度结果对比


##（PSNR值）（原文未给训练时的精度）

| platform | PSNR |
| -------- | ---- |
| gpu      | 18.7 |
| npu      | 19.2 |

# 推理结果对比( PSNR值)

| paltform/mask_type | center             | half               | random             | pattern            |
| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| 原文               | 19.0               | 14.6               | 33.0               | 19.8               |
| gpu                | 18.98              | 12.0               | 20.11              | 21.34              |
| npu                | 18.923322378004563 | 11.128324117994813 | 21.132619768680495 | 20.802661189895783 |
| 离线推理           | 15.526282777749453 | 11.542861669611257 | 17.84091132767341  | 17.11531867788335  |