<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：CV**

**版本（Version）：**

**修改时间（Modified） ：2022.05.05**

**框架（Framework）：TensorFlow 1.15.0**

**处理器（Processor）：昇腾910**

**描述（Description）：使用训练好的SVD模型，评估对称正交化在点云对准中的应用效果。** 

<h2 id="概述.md">概述</h2>

给定两个三维点云图，利用SVD正交化过程SVDO+(M)将其投射到SO(3)上，要求网络预测最佳对齐它们的3D旋转。代码的训练逻辑是每训练10w步保存一个模型，并且在测试集上检验该模型的精度，最后比较的都是260w步的模型精度

- 开源代码：

    https://github.com/google-research/google-research/tree/master/special_orthogonalization。

- 参考论文：

    [An Analysis of SVD for Deep Rotation Estimation](https://arxiv.org/abs/2006.14616)

- 参考实现：
	
	数据下载百度网盘链接：https://pan.baidu.com/s/1up1HW6McgSor3JF0yqQZSA 
提取码：2019
	
	共有3个数据集
	
	训练数据集points 
	
	测试数据集 points_test
	
	第一步旋转后的数据集 test_points_modified
	
	npu训练出来的模型下载百度网盘链接：https://pan.baidu.com/s/1JU1koZR7uGlkKfRYIk8tsw 
提取码：2019
	
	

	
- 相关迁移的工作：
  在进行代码迁移到NPU上时，输入的训练数据为点云数据，点云数据的shape为(N,3)，其中N并不是固定的，因此在NPU上存在动态shape的问题，导致模型训练无法正常进行。我们为此想了三个解决方法：1、找出所有点云数据中最小的N，对于大于N的点云数据，仅取前N行的数据输入训练。2、找到所有点云数据中最大的N，对于小于N的点云进行补0操作，将所有数据固定为最大的N后，输入网络进行训练。3、找到所有点云数据中最大的N，对小于N的点云数据，从原数据中选择一个点云进行填补至行数为N，再将数据输入网络进行训练。该三种方法均成功解决了NPU上的动态shape问题，但是第一种方法删除了样本点，因此导致最后训练出的模型精度很差；第二种方法虽然并没有丢失样本信息，但是向数据中填入大量的0，改变了本来的代码逻辑，导致最后训练出的模型精度也并不高。对于第三种方法，即没有丢失样本信息，对每个点云数据中的某一个点云样本点进行重复操作，没有改变原始的代码逻辑，最后也获得了不错的精度表现。

- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 数据集获取百度网盘链接：

- 训练超参

    - log_step_count=200 
    - save_summaries_steps=25000 
    - train_steps=2600000 
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
gen_pt_test_data_gpu.py 中的参数

--input_test_files 传入需要进行旋转的点云数据集
--output_directory 存储旋转后点云集的路径
--random_rotation_axang boole型，若为真将会对传入的数据集采用指定方法旋转，默认为真
--num_rotations_per_file  每个测试点云的随机旋转增加数。默认为100


main_point_cloud_gpu.py 中的参数

--pt_cloud_test_files  测试数据集路径 
--pt_cloud_train_files 熟练数据集路径 
--method   指定用于预测旋转的方式。选项为"svd", "svd-inf", or "gs"。默认为“svd”
--checkpoint_dir   训练模型的存放位置
--train_steps   训练迭代的次数。默认为2600000
--save_checkpoints_steps   保存检查点的频率。默认为10000
--log_step_count   日志记录一次的步数。默认为200
--save_summaries_steps   保存一次summary的步数。默认为5000
--learning_rate   默认为1e-5
--lr_decay   如果为真，则衰减learning rate。默认为假
--lr_decay_steps   learning rate衰减步数。默认为35000
--lr_decay_rate   learning rate衰减速率。默认为0.95
--predict_all_test   如果为真，则在最新的检查点上运行eval作业，并打印每个输入的误差信息。默认为假
--eval_examples   测试样本的数量。默认为0
--print_variable_names   打印模型变量名。默认为假
--num_train_augmentations   增加每个输入点云的随机旋转数。默认为10

```



## 运行

GPU运行命令如下：

**生成测试数据**

注：生成的文件test_points_modified、points已包含在dataset文件夹中。
```bash
# 将路径设置到训练点云图文件
IN_FILES=/points_test/*.pts

NEW_TEST_FILES_DIR=/test_points_modified

AXANG_SAMPLING=True

# 决定旋转轴角的分布
AXANG_SAMPLING=True

python -m special_orthogonalization.gen_pt_test_data_gpu --input_test_files=$IN_FILES --output_directory=$NEW_TEST_FILES_DIR --random_rotation_axang=$AXANG_SAMPLING
```

**训练与评价**
```bash
# 将路径设置到原始训练数据
TRAIN_FILES=/points/*.pts

#将路径设置到旋转后的训练数据
TEST_FILES=$NEW_TEST_FILES_DIR/*.pts

# 指定旋转预测方式
METHOD=svd

# 指定ckpt、summaries、评价结果等的存储路径
OUT_DIR=/path/to/model

python -m special_orthogonalization.main_point_cloud_gpu --method=$METHOD --checkpoint_dir=$OUT_DIR --log_step_count=200 --save_summaries_steps=25000 --pt_cloud_train_files=$TRAIN_FILES --pt_cloud_test_files=$TEST_FILES --train_steps=2600000 --save_checkpoints_steps=100000 --eval_examples=39900
```

**从所有训练样本中生成统计数据**
```bash
# 打印均值、中位数、标准差和分位数
python -m special_orthogonalization.main_point_cloud_gpu --method=$METHOD --checkpoint_dir=$OUT_DIR --pt_cloud_test_files=$TEST_FILES --predict_all_test=True
```
## 运行

NPU运行命令方式如下:

对于所有的三个步骤程序来说，modelarts插件obs桶中的数据路径均要写到真正包含数据的那一个路径
如在dataset文件夹中含有points、points_test等包含数据的文件夹
modelarts插件中的数据路径写为 obs://cann-id2019/dataset/

**生成测试数据**

运行这一步我们需要的程序文件为gen_pt_test_data.py、modelarts_entry_Axang.py、genTestData.sh
这三个文件中的代码均不需要修改
最后生成的旋转后的数据文件存放在obs桶当次程序文件的output路径中，文件名为test_points_modified，
为进行第二步模型训练，需要将生成旋转后的文件转移至obs桶中存放data的路径

注：需要确保的是存在obs桶里的data文件名为points_test

**训练与评价**

运行这一步我们需要的程序文件为main_point_cloud_boostPerf.py、modelarts_entry_acc_train.py、train_full_1p.sh
这三个文件中的代码均不需要修改

由于采用混合精度提高训练性能，一些算子计算溢出，为此增添switch_config.txt文件，该文件应该和代码所在目录一致。

最后生成的旋转后的数据文件存放在obs桶当次程序文件的output路径中，文件名为test_points_modified，
为进行第三步，需要将生成的output文件转移至obs桶中存放data的路径

注意：该次训练的模型保存在该次的obs文件夹中，进行第三步时又需要重启一次新的modelarts，因此我们需要将output文件中的
checkpoint文件中最新模型的路径修改
"/home/ma-user/modelarts/inputs/data_url_0/output"
这样第三步才能跑出正确的精度指标


**从所有训练样本中生成统计数据**

运行这一步我们需要的程序文件为main_point_cloud_boostPerf.py、modelarts_entry_stat.py、genStatistical.sh
这三个文件的代码均不需要修改

运行成功后将会在屏幕上打印出关于精度相应的统计量值



## 训练结果
**精度对比：**



|      测地线误差（°）        | 论文发布 | GPU(初始代码未改动版本) | GPU实测|NPU实测 |
| ------------------------ | ------- | ----- | --------- |----------|
|     平均值                |   1.63  |   2.58   |     3.98    |  2.92  |
|     中值                 |   0.89  |    1.68   |     2.6    |  1.7  |
|     标准差             |   6.70  |  6.93    |     9.36    |  8.45  |

相比于论文中的精度，我们NPU迁移后实测差距依然较大，但是与我们未对代码任何改动初始的版本在GPU上跑出来的精度相差较小，
且对于相同的代码的代码改动，NPU上的精度优于GPU上的精度。需要注意的是，在NPU上运行程序时，我们采用混合精度来提升训练
的性能，但是其中产生了未知的错误，导致代表的精度指标mean_degree_err在整个训练过程中始终为0，因此我们无法得知在NPU训练的
260w步的过程中，精度指标是下降的过程是怎样的。值得庆幸的是通过NPU训练出的模型，能够在GPU上计算出精度，并且精度还不错。

**性能对比：**

取华为v100上GPU运行的前2w步的global_step/sec平均值和NPU运行的前1w步的global_step/sec平均值进行对比，以达到性能对比的目的。
对于NPU上的性能计算 需要的程序为main_point_cloud_boostPerf.py、modelarts_entry_perf.py、train_performance_1p.sh，其中参数已经
设置完毕，无需更改。同时我们上传了我们计算性能的代码calc_perf.py，运行该代码需要将产生的日志文件从obs上下载下来，传入obs文件在本地的路径即可
|       性能指标项       | 论文发布 | GPU实测 | NPU实测 |
| ------------------- | ------- | ------ | ------  |
|     global_step/sec|    无    | 87.64  |   116.77  |

## 离线推理
参考 SVD_ID2019_for_ACL


