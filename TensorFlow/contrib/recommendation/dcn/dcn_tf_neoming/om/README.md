# dcn om文件说明
```bash
├── data                        #转成Bin文件的数据
│   ├── data                    	#每一个bin文件的形状是4000*39
│   └── label                   	#每一个bin文件的形状是4000*1
├── models                      # pb 和 om文件 其中om是Ascend910
│   ├── model.om					# atc命令转换得到的om
│   ├── model.pb					# session文件夹下的ckpt转换得到的pb
│   └── session						# 通过keras.backend.get_session()得到的ckpt
├── msame                       # 在裸机上编译的msame工具
├── om_output                   # msame工具相关输出
│   └── 20210723_192051				# msame输出的bin文件
├── README.md					# 本文档
└── scripts						# 脚本
    ├── check_data.py				# 检查数据
    ├── ckpt2pb.py					# ckpt转换成pb的文件
    ├── data2bin.py					# 数据从numpy转成bin文件
    ├── eval.py						# 验证om输出的精度
    ├── export.sh					# 通过atc命令将pb转成om
    └── inference.sh				# 使用msame进行推理

```



## 2如何运行脚本

运行前需要更改脚本中的路径

### 2.1如何通过pb导出om

```bash
cd scripts
./export.sh
```

### 2.2如何使用om进行推理

```bash
cd scripts
./inference.sh
```

## 如何验证om输出的精度

auc与在线推理一致

```bash
cd scripts
python3 eval.py 
AUC-ROC: 0.8052, AUC-PR: 0.5975
```

