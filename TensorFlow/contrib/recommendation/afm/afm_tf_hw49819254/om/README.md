# AFM 离线推理

### 提交文件夹说明：

```shell
├── atc_output                                  # atc命令的输出
│   ├── fusion_result.json                      
│   └── kernel_meta
├── data                                        # om使用的相关数据
│   ├── afm_output_0.bin                        # msame得到的输出文件
│   ├── performance.jpg                         # 原ckpt对应的精度
│   ├── dropout_keep_afm.bin                    # 测试数据的bin文件
│   ├── train_features_afm.bin   		# 测试数据的bin文件
│   ├── train_labels_afm.bin   			# 测试数据的bin文件
│   ├── train_phase_afm.bin   			# 测试数据的bin文件
│   └── y_true.bin                          	# 测试标签的bin文件
├── model                                       # 模型文件
│   └── download_link.png                       # pb和om文件下载链接
├── msame                                       # msame离线推理工具
├── out                                   	# msame得到的输出文件夹
│   └── 20210607_182502                         # 输出文件夹
│       └── afm_output_0.bin                	# 模型得到的输出
├── README.md                                   # 本README文件
└── scripts                                     # 相关脚本
    ├── eval.py                                 # 验证om模型输出结果的精度
    ├── export.sh                               # 将pb转换成om的shell脚本
    └── inference.sh                            # 把输入数据喂给om模型的推理脚本
```