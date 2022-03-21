## PIN：离线推理

### 提交文件夹说明：

```
├── atc_output                                  # atc命令的输出
│   ├── fusion_result.json                      
│   └── kernel_meta
├── data                                        # om使用的相关数据
│   ├── inputs                         			# 测试数据的bin文件夹
│   └── labels                          		# 测试标签的bin文件夹
├── models                                      # 模型文件
│   ├── model.om                              	# model.om文件
│   └── model.pb                              	# model.pb文件
├── msame                                       # msame离线推理工具
├── om_output                                   # msame得到的输出文件夹
│   └── 20210708_221703                        	# 输出文件夹
├── README.md                                   # 本README文件
└── scripts                                     # 相关脚本
    ├── export.sh                               # 将pb转换成om的shell脚本
    ├── inference.sh                            # 把输入数据喂给om模型的推理脚本
    └── om_eval.py                              # 验证om模型输出结果的精度
```



### 相关说明

#### 样本评估

- 原本完整的test数据集以2000为batch_size可以取2050个输入和输出bin文件（总数并不为2000的整倍数），但在模型中为了规避动态shape从而将input的形状固定成2000x16，从而使得第2050个输入bin文件不满足输入形状而无法输入om。因此将2050.bin文件舍去。
- 在接下来的om_eval中，发现有一个om输出的预测值少1，使得score的总数量比test_label的数值少1，通过检测得到该文件为999.bin，从而将该文件剔除，因此最终检测的test样本数为2048。

#### bin文件

- [inputs和lables](https://gitee.com/Apochen/i-pin-you)
- 将上述repo中的bin目录下的inputs和labels目录及目录下的文件拷贝到data文件夹下，带时间戳的bin文件目录拷贝到om_output文件夹下

#### om和pb文件

- 在bin文件获取仓库内的models目录下有对应的pb和om文件。