<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：CV**

**版本（Version）：**

**修改时间（Modified） ：2022.3.25**

**框架（Framework）：TensorFlow 1.15.0**

**处理器（Processor）：昇腾910**

**描述（Description）：使用训练好的SVD模型，评估对称正交化在点云对准中的应用效果。** 

<h2 id="概述.md">概述</h2>

给定两个三维点云图，利用SVD正交化过程SVDO+(M)将其投射到SO(3)上，要求网络预测最佳对齐它们的3D旋转。

- 开源代码：

    https://github.com/google-research/google-research/tree/master/special_orthogonalization。

- 参考论文：

    [An Analysis of SVD for Deep Rotation Estimation](https://arxiv.org/abs/2006.14616)

- 参考实现：

    obs://cann-id2019/gpu/


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 数据集获取：包含在dataset里，需要解压缩。

- 训练超参

    - log_step_count=200 
    - save_summaries_steps=25000 
    - train_steps=2200000 
    - save_checkpoints_steps=100000
    - eval_examples=39900

    

<h2 id="训练环境准备.md">训练环境准备</h2>

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>



## 脚本参数<a name="section6669162441511"></a>

```
--input_test_files1  原始输入测试文件的正则表达式 
--input_test_files2  原始输入测试文件的正则表达式 
--output_directory1   将存储新测试文件的输出目录
--output_directory2   将存储新测试文件的输出目录
--num_rotations_per_file  每个测试点云的随机旋转增加数。默认为100
--random_rotation_axang   如果为真，则使用该方法从原始基准代码中对随机旋转进行采样，否则样本采用哈尔测量。默认为真
--method   指定用于预测旋转的方式。选项为"svd", "svd-inf", or "gs"。默认为“svd”
--checkpoint_dir   保存检查点位置
--train_steps   训练迭代的次数。默认为2600000
--save_checkpoints_steps   保存检查点的频率。默认为10000
--log_step_count   日志记录一次的步数。默认为500
--save_summaries_steps   保存一次summary的步数。默认为5000
--learning_rate   默认为1e-5
--lr_decay   如果为真，则衰减learning rate。默认为假
--lr_decay_steps   learning rate衰减步数。默认为35000
--lr_decay_rate   learning rate衰减速率。默认为0.95
--predict_all_test   如果为真，则在最新的检查点上运行eval作业，并打印每个输入的错误。默认为假
--eval_examples   测试样本的数量。默认为0
--print_variable_names   打印模型变量名。默认为假
--num_train_augmentations   增加每个输入点云的随机旋转数。默认为10
--pt_cloud_train_files   匹配所有训练点文件的表达式
--pt_cloud_test_files   匹配所有修改的测试点文件的表达式

```



## 运行

GPU运行命令如下：

**修改原始测试数据**

注：生成的文件points_test_modified、points0已包含在dataset文件夹中。
```bash
# 将路径设置到训练点云图文件
IN_FILES1=/shapenet/data/pc_plane/points/*.pts
IN_FILES2=/shapenet/data/pc_plane/points_test/*.pts

# 设置新生成文件的路径
NEW_TEST_FILES_DIR1=/shapenet/data/pc_plane/points0
NEW_TEST_FILES_DIR2=/shapenet/data/pc_plane/points_test_modified

# 决定旋转轴角的分布
AXANG_SAMPLING=True

python -m special_orthogonalization.gen_pt_test_data --input_test_files1=$IN_FILES1 --input_test_files2=$IN_FILES2 --output_directory1=$NEW_TEST_FILES_DIR1 --output_directory2=$NEW_TEST_FILES_DIR2 --random_rotation_axang=$AXANG_SAMPLING
```

**训练与评价**
```bash
# 将路径设置到原始训练数据
TRAIN_FILES=/shapenet/data/pc_plane/points0/*.pts

#将路径设置到旋转后的训练数据
TEST_FILES=$NEW_TEST_FILES_DIR2/*.pts

# 指定旋转预测方式
METHOD=svd

# 指定ckpt、summaries、评价结果等的存储路径
OUT_DIR=/path/to/model

python -m special_orthogonalization.main_point_cloud --method=$METHOD --checkpoint_dir=$OUT_DIR --log_step_count=200 --save_summaries_steps=25000 --pt_cloud_train_files=$TRAIN_FILES --pt_cloud_test_files=$TEST_FILES --train_steps=2200000 --save_checkpoints_steps=100000 --eval_examples=39900
```

**从所有训练样本中生成统计数据**
```bash
# 打印均值、中位数、标准差和分位数
python -m special_orthogonalization.main_point_cloud --method=$METHOD --checkpoint_dir=$OUT_DIR --pt_cloud_test_files=$TEST_FILES --predict_all_test=True
```

## 训练结果
**精度对比：**

由于弹性云服务器上Tesla V100的GPU训练环境，选用矩池云GPU的运行结果进行精度对比。


|      测地线误差（°）        | 论文发布 | GPU实测 | NPU实测 |
| ------------------------ | ------- | ----- | ------- |
|     平均值                |   1.63  |   5.55    |     待测    |
|     中值                 |   0.89  |    3.65   |     待测    |
|     标准偏差             |   6.70  |  10.68     |     待测    |

**性能对比：**

取弹性云GPU运行的前2600步的global_step/sec平均值和NPU运行的前2600步的global_step/sec平均值进行对比，以达到性能对比的目的。

|       性能指标项       | 论文发布 | GPU实测 | NPU实测 |
| ------------------- | ------- | ------ | ------  |
|     global_step/sec|    无    | 79.48  |   66.08  |


