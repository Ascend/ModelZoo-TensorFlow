# DevNet: 离线推理

## 提交文件夹说明：
```shell
├── atc_output                                  # atc命令的输出
│   ├── fusion_result.json                      
│   └── kernel_meta
├── data                                        # om使用的相关数据
│   ├── dev-net_output_0.bin                    # msame得到的输出文件
│   ├── performance.txt                         # 原ckpt对应的精度
│   ├── test_input.bin                          # 测试数据的bin文件
│   └── test_label.bin                          # 测试标签的bin文件
├── models                                      # 模型文件
│   ├── dev-net.om                              # dev-net.om文件
│   └── dev-net.pb                              # dev-net.pb文件
├── msame                                       # msame离线推理工具
├── om_output                                   # msame得到的输出文件夹
│   └── 20210413_204135                         # 输出文件夹
│       └── dev-net_output_0.bin                # 模型得到的输出
├── README.md                                   # 本README文件
└── scripts                                     # 相关脚本
    ├── eval.py                                 # 验证om模型输出结果的精度
    ├── export.sh                               # 将pb转换成om的shell脚本
    └── inference.sh                            # 把输入数据喂给om模型的推理脚本
```
