## 概述

Deep-SLR是一个磁共振图像快速重建网络。它将结构低秩先验信息与卷积神经网络相结合，摆脱了在无校准传统磁共振快速重建算法中的高计算复杂度，和有校准传统磁共振快速重建算法中对于预校准全采样区域的依赖，使用机器学习估计映射关系，实现较高速的磁共振图像重建，减少了图像恢复时间、提高了图像质量。

### 参考论文

> Pramanik, Aniket, Hemant Aggarwal, and Mathews Jacob, "Deep Generalization of Structured Low-Rank Algorithms (Deep-SLR)", IEEE Transactions on Medical Imaging, 2020. https://ieeexplore.ieee.org/document/9159672
>
> Pramanik, Aniket, Hemant Aggarwal, and Mathews Jacob, "Calibrationless Parallel MRI using Model Based Deep Learning (C-MODL)", 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI). https://ieeexplore.ieee.org/document/9098490

## 数据要求

本模型使用 Raw Data，并需要模板（mask）进行模拟欠采样操作，需要用户自行获取。

提供的示例数据可在[原GitHub项目](https://github.com/anikpram/Deep-SLR)说明中自行获取。

示例文件为：“vardenmask_6f.mat”，“trn_data_90im_4_subjects.npy”，“tst_img.npy”，请自行核对。

## 环境配置

* 有关环境配置可在 “npu_train.sh”中进行修改。其中，将数据存放目录与 `data_path` 对应，将结果保存目录与 `output_path` 对应。
* 所需第三方包可在“requirements.txt”中进行核对。特别地，如果`scikit-image`包版本为 0.18.x，则需要将“auxiliaryFunctions.py”中的导入部分按照提示进行修改，修改说明在代码注释中已经给出。
  如果使用低于 0.18.0 的版本，则无须处置。

## 精度指标

下表给出了论文中、GPU上以及NPU上的测试精度，使用了示例数据。

|      |  论文精度   | GPU精度 | **NPU精度** |
| ---- | :---------: | :-----: | :---------: |
| PSNR | 34.46±1.22  | 34.361  | **34.445**  |
| SSIM | 0.958±0.011 |  0.921  |  **0.924**  |

## 性能指标

GPU性能：perf_mean: 17.0435ms/step
NPU性能: perf_mean: 0.9084ms/step  fps_mean:1112.7464

## 复现步骤

1. 使用pip和“requirements.txt”文件安装第三方库。
1. 将训练用数据、测试用数据和模板放在所指定的 `data_path` 下。
1. 从“modelarts_entry.py”进入，运行模型进行训练。训练完成后模型会自动进行测试。测试结果会产生标准输出、以图像形式显示（仅 GUI）并保存在 `output_path` 下，生成“test_res.txt”和“test_res.png”。其中的三个数字与图像中显示的三个数据一一对应，分别为输入（测试）图像 PSNR，重建图像 PSNR，重建图像 SSIM。

## 样例文件

请在[此](https://wwo.lanzoui.com/ioUKQwq99hi)下载，其中提供了预训练模型及其测试结果。