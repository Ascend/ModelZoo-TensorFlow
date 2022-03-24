# SEQUENCE_TAGGING离线推理

## 环境要求

| 环境 | 版本 |
| --- | --- |
| CANN | >=5.0.3 |
| 处理器| Ascend310/Ascend910 |
| 其他| 见 'requirements.txt' |

## 数据准备
1. 模型训练使用CoNLL2003数据集，数据集请用户自行获取。

2. 数据集训练前需要做预处理操作，请参考默认配置中的数据集预处理小结。

3. 数据集处理后，放入SEQUENCE_TAGGING_ID2097_for_ACL/data目录下。

## 数据集预处理（CoNLL2003语料库）：

- 去除数据集中与命名实体识别无关的属性（第2、3列）和DOCSTART行  
    处理前：

    ```text
    -DOCSTART- -X- -X- O
    
    EU NNP B-NP B-ORG
    rejects VBZ B-VP O
    German JJ B-NP B-MISC
    call NN I-NP O
    to TO B-VP O
    boycott VB I-VP O
    British JJ B-NP B-MISC
    lamb NN I-NP O
    . . O O
    ```
    处理后：
    ```text
    EU B-ORG
    rejects O
    German B-MISC
    call O
    to O
    boycott O
    British B-MISC
    lamb O
    . O
    ```

- 数据集文件路径 

    训练集：./data/coNLL/eng/eng.train.iob   

    测试集：./data/coNLL/eng/eng.testb.iob   

    验证集：./data/coNLL/eng/eng.testa.iob  

- 词向量库预处理：

  - glove.6B下载

  - 词向量库文件路径      
    
    ./data/glove.6B

- 注意，推理所用的单词表必须与训练所用的一致，训练所需的单词表构建见SEQUENCE_TAGGING_ID2097_for_Tensorflow/build_data.py。

    数据集链接：[OBS](obs://cann-id2097/dataset/)
## 脚本和示例代码

```text
├── build_data.py                             //创建单词表
├── README.md                                 //代码说明文档
├── eval_ckpt.py                              //评估ckpt
├── eval_pb.py                                //评估pb
├── eval_om.py                                //评估om
├── convert_bin.py                            //数据转换
├── freeze_graph.py                           //模型固化
├── requirements.txt                          //环境依赖
├── LICENSE.txt                               //证书
├── scripts
│    ├──pb_to_om.sh                         //pb转om
│    ├──run_msame.sh                        //msame离线推理
├── model                                     
│    ├──__init__.py
│    ├──base_model.py                        //基础模型
│    ├──ner_model.py                         //网络结构
│    ├──config.py                            //参数设置
│    ├──data_utils.py                        //数据集处理
├── om_model                                 //存放om模型
├── pb_model                                 //存放pb模型
├── bin_data                                 //存放bin文件                          
```

## 模型文件

包括ckpt、pb、om模型文件

下载链接：[OBS](obs://cann-id2097/npu/Inference/)

## STEP1: ckpt文件转pb模型

```bash
# CKPT_PATH为ckpt文件的路径
python3 freeze_graph.py --dir_ckpt CKPT_PATH
# 示例
python3 freeze_graph.py --dir_ckpt ./ckpt/model.weights/
```

## STEP2: pb模型转om模型

检查环境中ATC工具环境变量，设置完成后，修改PB和OM文件路径PB_PATH和OM_PATH，运行pb_to_om.sh

```bash
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

PB_PATH=/root/infer
OM_PATH=/root/infer

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$PB_PATH/SEQUENCE_TAGGING.pb --framework=3 \
        --output=$OM_PATH/SEQUENCE_TAGGING --soc_version=Ascend310 \
        --input_shape="word_ids:1,128;sequence_lengths:1;char_ids:1,128,64" \
        --out_nodes="dense/BiasAdd:0;transitions:0"

```

## STEP3: 数据集转为bin文件

运行convert_bin.py

```bash
# DATA_PATH为数据集路径
python3 convert_bin.py --data_path DATA_PATH
# 示例
python3 convert_bin.py --data_path ./data
```

## STEP4: 使用msame工具推理

安装好[msame]([tools: Ascend tools - Gitee.com](https://gitee.com/ascend/tools/tree/master/msame))，运行run_msame.sh

```bash
OM_PATH=/root/infer
BIN_PATH=/root/infer/bin_data

# /root/msame/out/改成自己的msame安装路径
/root/msame/out/./msame --model $OM_PATH/SEQUENCE_TAGGING.om --input $BIN_PATH/word_ids,$BIN_PATH/sequence_lengths,$BIN_PATH/char_ids --output $OM_PATH/ 
```

注意，msame生成的推理文件夹是根据时间命名的，类似于20220323_170719这样的格式，需要自己检查路径，在后续精度验证的步骤中修改。SEQUENCE_TAGGING模型的输出有两个，需要将这两个输出分别存放到两个文件夹dir_om_output/output_0 和 dir_om_output/output_1（dir_om_output需要改为自己创建的文件夹）

## 验证om模型精度

运行eval_om.py。

```bash
# DATA_PATH为数据集路径，OM_OUTPUT为om模型推理的输出（bin格式）
python3 eval_om.py --data_path DATA_PATH --dir_om_output OM_OUTPUT
# 示例
python3 eval_om.py --data_path ./data --dir_om_output ./bin_data
```

## 验证pb模型精度

运行eval_pb.py。

```bash
# DATA_PATH为数据集路径，CKPT_PATH为ckpt文件路径
python3 eval_pb.py --data_path DATA_PATH
# 示例
python3 eval_pb.py --data_path ./data
```

## 验证ckpt精度

运行eval_ckpt.py。

```bash
# DATA_PATH为数据集路径，CKPT_PATH为ckpt文件路径
python3 eval_ckpt.py --data_path DATA_PATH --dir_ckpt CKPT_PATH
# 示例
python3 eval_ckpt.py --data_path ./data --dir_ckpt ./ckpt/model.weights
```