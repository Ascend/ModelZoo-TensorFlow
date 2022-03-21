# HAN离线推理



## 提交文件夹说明

```shell
├── data																								# om的input数据
│   ├── 1.bin
│   ├── 2.bin
│   ├── 4.bin
│   └── 5.bin
├── HAN_final.om																			# HAN.om文件
├── msame
├── output
│   └── 20210603_131312
│       └── HAN_output_0.bin														# 模型得到的输出
└── README.md
```

验证om精度的文件在项目主目录下acc_compare.py，om模型下的准确度为86.64%

![](../pic/om_acc.png)
