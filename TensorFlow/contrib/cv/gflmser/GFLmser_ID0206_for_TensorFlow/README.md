### 项目概述

本项目复现 [GFLmser](https://dl.acm.org/doi/10.1145/3343031.3350952) 模型，项目代码基于 TensorFlow 框架，可运行于 Ascend 910，精度比对结果如下表：

| ---- | 原论文 | Ascend 910 | NVIDIA TITAN Xp |
| ---- | :----: | :--------: | :-------------: |
| PSNR | 26.76  |      |  23.43   |

###  目录结构

```
gflmser
├── LICENSE
├── README.md
├── data                # 数据集
│   ├── test
│   └── train
├── GFLmser.py          # 模型代码
├── data.py             # 加载数据代码
├── main_li.py          # 训练与测试代码
└── train_testcase.sh
```

### 数据集说明

data 目录下需要包含 train、test两个子目录，分别对应训练集、测试集。每个子目录下是若干个 npy 文件，包含经过预处理的数据，数据以字典格式存储，包含两个键：

* sample：低分辨率图像
* label：高分辨率图像

[数据文件](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=QjOtwR3TTfSsZB08dJ7o4Ab26ZH0vHZj9C/n+HEnR++XEHML4FSGXzhmtl5lwrfyC2P+2lCIy33fO4uWaMcQ1zR2aR5ddB3uDkSJ7NpPvwgUt6cWIsJvEfmnvwsuc1PRoWI+5TD+Pn+CuQoD6WU5TfrRIpS/WlthsXSwWuaZwmST+yhzTIaCD/3lspkG4TQHn/kjN7eirYmFHUE/i/Ma191NfxElJP2n184GECCL5em+VArg7BKK+lE9fsE5AGZmq6kPAo5VwYk/EJdXvYsz9bysdpmruJFnytAfs1r4zfiZVdBULKjMsZj9SSus2FjVQmkcg46YPRjeFMY+JcTeeEvSg93E3XjCicTaghXvwkCvq9Ct3CiP9MucaV97kJDf8NQ9IiBCJ3/WZjfa+fPuL7aPjU2LLfFoREjD6e+hdpwdHtBLJz4OhiX6JBwPsD8yaVkz9+MbvD42X0IZQqLCbdiJ1spmouCJFW6f3UsFcEB2c0ECwTYAmQpr3mAnivpw2Y2gBTmTgyWI5dvp/C4nMw==)
提取码：111111


### 训练

* 下载数据集，放置在 data 目录下
* 将train_url 和 data_url 设置为OBS上的相应路径
* 运行main_li.py
