-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [推理准备](#推理准备.md)
-   [快速上手](#快速上手.md)
-   [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Human body 3D reconstruction 

**版本（Version）：1.2**

**修改时间（Modified） ：2021.11.5**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt, pb, om**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910A, 昇腾310**

**应用级别（Categories）：Demo**

**描述（Description）：基于TensorFlow框架的HMR人体三维重建网络推理代码** 

<h2 id="概述.md">概述</h2>

- HMR是一种人体三维重建算法，通过采用2D和3D形式的监督，并引入生成对抗网络修正输出结果的分布来实现从单张图像恢复人体的三维模型。

- 模型架构

  ![overview](overview.png)

- 参考论文：

    [Kanazawa A, Black M J, Jacobs D W, et al. End-to-end recovery of human shape and pose[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7122-7131.](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.pdf)

- [参考实现](https://github.com/akanazawa/hmr)

<h2 id="推理准备.md">推理准备</h2>

- atc工具

- msame工具

- 根据requirements.txt配置所需依赖。

- 数据集[下载](https://disk.pku.edu.cn:443/link/4213EF310253E1C75B96CEB3B6C89135)

- 预训练模型[下载](https://disk.pku.edu.cn:443/link/215748269F58022E672DCAFD661DD251)

<h2 id="快速上手.md">快速上手</h2>

1. 在脚本`scripts/convert_test_data.sh`中，配置路径，运行脚本对原始的TFRecord格式的测试数据进行转换，得到bin格式的输入和npy格式的GT。

   ```
   bash scripts/convert_test_data.sh
   ```

2. 在脚本`scripts/freeze_graph.sh`中，配置路径，运行脚本对ckpt模型进行固化，得到pb模型。

   ```
   bash scripts/freeze_graph.sh
   ```

3. 在脚本`scripts/inference_from_pb.sh`中，配置路径，运行脚本使用pb模型进行推理，验证2中模型固化的正确性。

   ```
   bash scripts/inference_from_pb.sh
   ```

4. 在脚本`scripts/pb2om.sh`中，配置路径，运行脚本对pb模型进行转换，得到om模型。

   ```
   bash scripts/pb2om.sh
   ```

5. 在脚本`scripts/inference_from_om.sh`中，配置路径，运行脚本使用om模型进行推理，得到输出。

   ```
   bash scripts/inference_from_om.sh
   ```

3. 在脚本`scripts/cal_om_metrics.sh`中，配置路径，运行脚本计算om推理指标，验证4、5中模型转换、推理的正确性。

   ```
   bash scripts/cal_om_metrics.sh
   ```

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
|-- LICENSE
|-- README.md
|-- overview.png
|-- requirements.txt
|-- scripts
|   |-- cal_om_metrics.sh
|   |-- convert_test_data.sh
|   |-- freeze_graph.sh
|   |-- inference_from_om.sh
|   |-- inference_from_pb.sh
|   `-- pb2om.sh
`-- src
    |-- benchmark
    |   |-- __init__.py
    |   `-- eval_util.py
    |-- cal_om_metrics.py
    |-- config.py
    |-- convert_test_data.py
    |-- datasets
    |   |-- __init__.py
    |   `-- common.py
    |-- freeze_graph.py
    |-- inference_from_pb.py
    |-- models.py
    `-- tf_smpl
        |-- LICENSE
        |-- __init__.py
        |-- batch_lbs.py
        |-- batch_smpl.py
        `-- projection.py
```