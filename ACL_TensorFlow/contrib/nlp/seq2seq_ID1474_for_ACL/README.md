# Seq2Seq

Seq2Seq推理部分实现，模型概述详情请看Seq2Seq_ID1474_for_TensorFlow README.md

## 训练环境

- TensorFlow 1.15.0
- Python 3.7.0

## 代码及路径解释

```
seq2seq_ID1474_for_ACL
│  compute_bleu.py			bleu计算
│  convert_bin.py			原始数据转bin文件
│  data_utils.py			原始数据处理
│  evaluate.py				验证om模型推理精度
│  freeze_graph.py			ckpt模型转pb模型
│  main						msame编译后文件
│  README.md				
│  run_msame.sh				msame离线推理
│  run_om.sh				pb模型转om模型
│  seq2seq_model.py			seq2seq模型文件
├─bin_data					存放bin格式的数据文件
├─data						存放原始数据文件
├─model						存放ckpt模型文件
├─msame_out  				存放msame推理结果文件
├─om_model 					存放om模型文件
├─override_contrib			seq2seq搭建模型所需脚本文件
│      core_rnn_cell.py
│      override_seq2seq.py
└─pb_model					存放pb模型文件	            
```

## 数据集

- newstest2013.en
- newstest2013.fr
- vocab80000.to
- vocab160000.from

下载链接：[Link(dmns)](https://pan.baidu.com/s/1Gu_CjILJH-5N2IND9kPF4Q)

## 模型文件

包括ckpt、pb、om模型文件

下载链接：[Link(dmns)](https://pan.baidu.com/s/1Gu_CjILJH-5N2IND9kPF4Q)

## pb模型

./model文件夹中存放ckpt模型，运行frzee_graph.py

```
python3 freeze_graph.py
```

## 生成om模型

检查环境中ATC工具环境变量，设置完成后，修改PB和OM文件路径PB_DIR和OM_DIR，运行run_om.sh

```
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

PB_DIR=/home/test_user02/th/seq2seq_ID1474_for_ACL/pb_model
OM_DIR=/home/test_user02/th/seq2seq_ID1474_for_ACL/om_model

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$PB_DIR/seq2seq.pb \
        --framework=3 \
        --output=$OM_DIR/seq2seq \
        --soc_version=Ascend910 \
        --input_shape="encoder0:1;encoder1:1;encoder2:1;encoder3:1;encoder4:1;encoder5:1;encoder6:1;encoder7:1;encoder8:1;encoder9:1;encoder10:1;encoder11:1;encoder12:1;encoder13:1;encoder14:1;encoder15:1;encoder16:1;encoder17:1;encoder18:1;encoder19:1;encoder20:1;encoder21:1;encoder22:1;encoder23:1;encoder24:1;encoder25:1;encoder26:1;encoder27:1;encoder28:1;encoder29:1;encoder30:1;encoder31:1;encoder32:1;encoder33:1;encoder34:1;encoder35:1;encoder36:1;encoder37:1;encoder38:1;encoder39:1;decoder0:1" \
        --log=info

```

## 将数据集转为bin文件

./data文件夹中存放数据集，运行convert_bin.py

```
python3 convert_bin.py
```

## 使用msame工具推理

安装好[msame]([tools: Ascend tools - Gitee.com](https://gitee.com/ascend/tools/tree/master/msame))，修改msame安装路径，BASE_DIR后，运行run_msame.sh

```
BASE_DIR=/home/test_user02/th/seq2seq_ID1474_for_ACL
BIN_DATA_DIR=$BASE_DIR/bin_data

encoder0_path=$BIN_DATA_DIR/encoder0
encoder1_path=$BIN_DATA_DIR/encoder1
encoder2_path=$BIN_DATA_DIR/encoder2
encoder3_path=$BIN_DATA_DIR/encoder3
encoder4_path=$BIN_DATA_DIR/encoder4
encoder5_path=$BIN_DATA_DIR/encoder5
encoder6_path=$BIN_DATA_DIR/encoder6
encoder7_path=$BIN_DATA_DIR/encoder7
encoder8_path=$BIN_DATA_DIR/encoder8
encoder9_path=$BIN_DATA_DIR/encoder9
encoder10_path=$BIN_DATA_DIR/encoder10
encoder11_path=$BIN_DATA_DIR/encoder11
encoder12_path=$BIN_DATA_DIR/encoder12
encoder13_path=$BIN_DATA_DIR/encoder13
encoder14_path=$BIN_DATA_DIR/encoder14
encoder15_path=$BIN_DATA_DIR/encoder15
encoder16_path=$BIN_DATA_DIR/encoder16
encoder17_path=$BIN_DATA_DIR/encoder17
encoder18_path=$BIN_DATA_DIR/encoder18
encoder19_path=$BIN_DATA_DIR/encoder19
encoder20_path=$BIN_DATA_DIR/encoder20
encoder21_path=$BIN_DATA_DIR/encoder21
encoder22_path=$BIN_DATA_DIR/encoder22
encoder23_path=$BIN_DATA_DIR/encoder23
encoder24_path=$BIN_DATA_DIR/encoder24
encoder25_path=$BIN_DATA_DIR/encoder25
encoder26_path=$BIN_DATA_DIR/encoder26
encoder27_path=$BIN_DATA_DIR/encoder27
encoder28_path=$BIN_DATA_DIR/encoder28
encoder29_path=$BIN_DATA_DIR/encoder29
encoder30_path=$BIN_DATA_DIR/encoder30
encoder31_path=$BIN_DATA_DIR/encoder31
encoder32_path=$BIN_DATA_DIR/encoder32
encoder33_path=$BIN_DATA_DIR/encoder33
encoder34_path=$BIN_DATA_DIR/encoder34
encoder35_path=$BIN_DATA_DIR/encoder35
encoder36_path=$BIN_DATA_DIR/encoder36
encoder37_path=$BIN_DATA_DIR/encoder37
encoder38_path=$BIN_DATA_DIR/encoder38
encoder39_path=$BIN_DATA_DIR/encoder39
decoder0_path=$BIN_DATA_DIR/decoder0

ulimit -c 0
/home/test_user02/th/seq2seq_ID1474_for_ACL/main --model $BASE_DIR/om_model/seq2seq.om \
  --input ${encoder0_path},${encoder1_path},${encoder2_path},${encoder3_path},${encoder4_path},${encoder5_path},${encoder6_path},${encoder7_path},${encoder8_path},${encoder9_path},${encoder10_path},${encoder11_path},${encoder12_path},${encoder13_path},${encoder14_path},${encoder15_path},${encoder16_path},${encoder17_path},${encoder18_path},${encoder19_path},${encoder20_path},${encoder21_path},${encoder22_path},${encoder23_path},${encoder24_path},${encoder25_path},${encoder26_path},${encoder27_path},${encoder28_path},${encoder29_path},${encoder30_path},${encoder31_path},${encoder32_path},${encoder33_path},${encoder34_path},${encoder35_path},${encoder36_path},${encoder37_path},${encoder38_path},${encoder39_path},${decoder0_path} \
  --output $BASE_DIR/msame_out/ \
  --outfmt TXT \
  --device 1
```

注意，msame生成的推理文件夹是根据时间命名的，类似于20211226_10_56_39_215427这样的格式，需要自己检查路径，在后续精度验证的步骤中修改。

## 使用推理得到的txt文件进行推理（om模型推理）

修改推理文件路径msame_out_dir，数据集文件路径data_dir。

```
python3 evaluate.py
```

## 使用原始数据集进行ckpt模型推理

修改数据集文件路径data_dir，ckpt模型文件路径train_dir。

```
python3 evaluate_ckpt.py
```

## 精度

BLEU-Score

- ckpt模型推理：14.23%
- om模型推理：13.79%

## 推理文本

- input

![](https://images.gitee.com/uploads/images/2022/0107/214124_dfbb125b_5559452.png)

- output

![](https://images.gitee.com/uploads/images/2022/0107/214307_b7848e42_5559452.png)

## 推理性能

```
Inference time: 153.776ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
Inference average time : 154.47 ms
Inference average time without first time: 154.46 ms
```