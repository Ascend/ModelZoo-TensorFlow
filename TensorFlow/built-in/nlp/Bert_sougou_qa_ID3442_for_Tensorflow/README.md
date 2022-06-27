本项目利用keras-bert和tokenizers模块，对BERT进行微调，实现抽取式问答（阅读理解的一种形式）。

### 维护者

- Jclian91

### 代码结构

项目结构如下：

```
.
├── bert_sougou_qa.h5（训练后的模型）
├── chinese_L-12_H-768_A-12（BERT-base中文预训练模型）
│   ├── bert_config.json
│   ├── bert_model.ckpt.data-00000-of-00001
│   ├── bert_model.ckpt.index
│   ├── bert_model.ckpt.meta
│   ├── test.json
│   └── vocab.txt
├── data（sougou问答数据集）
│   ├── sogou_qa_test.json
│   ├── sogou_qa_train.json
│   ├── test.json
├── evaluate_tool（sougou问答自带的评估工具）
│   ├── evaluate.py
│   ├── punctuation
│   ├── qa_pred
│   ├── qa_result
│   ├── qid_answer_expand
│   ├── README.md
│   ├── string_tool.py
├── keras_bert_model_evaluate.py（模型评估脚本）
├── keras_bert_model_predict.py（模型预测脚本）
├── keras_bert_model_server.py（模型服务脚本）
├── keras_bert_model_train.py（模型训练脚本）
└── requirements.txt
```


### 模型评估

exact match score: 0.6875

利用sougou问答自带的评估工具（使用Python2运行，运行方式参考网址：https://github.com/bojone/dgcnn_for_reading_comprehension/tree/master/evaluate_tool），评估结果如下：

```
[0.720125786163522, 0.8169927616633956, 0.7685592739134588]
```

上述结果的意思为：Accuracy=0.720125786163522，F1=0.8169927616633956，Final=0.7685592739134588

### 模型服务调用

启动脚本`keras_bert_model_server`，调用接口的curl命令如下：

```bash
curl --location --request POST 'http://192.168.1.193:16500/model/bert_qa' \
--header 'Content-Type: application/json' \
--data-raw '{
    "context": "中国最冷小镇是大兴安岭呼中区。据了解，呼中区位于大兴安岭伊勒呼里山脉北麓，年平均气温达-4.3℃，城镇历史最低温度达-53.2℃，有中国最冷小镇之称，每年这里下霜降雪较其他地区较早。 2020年10月11日早4时该地迎来了今年第一场大范围降雪天气，...",
    "question": "中国最冷小镇在哪里？"
}'
```

输出结果:

```
{
    "code": 200,
    "data": {
        "answer": "大兴安岭呼中区",
        "confidence": 0.8846356868743896,
        "context": "中国最冷小镇是大兴安岭呼中区。据了解，呼中区位于大兴安岭伊勒呼里山脉北麓，年平均气温达-4.3℃，城镇历史最低温度达-53.2℃，有中国最冷小镇之称，每年这里下霜降雪较其他地区较早。 2020年10月11日早4时该地迎来了今年第一场大范围降雪天气，...",
        "question": "中国最冷小镇在哪里？"
    },
    "message": "success"
}
```