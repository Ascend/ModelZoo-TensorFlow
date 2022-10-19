中文|[English](README_EN.md)

# Wide&Deep TensorFlow离线推理 

此链接提供Wide&Deep TensorFlow模型在NPU上离线推理的脚本和方法。

## 注意
**此案例仅为您学习Ascend软件栈提供参考，不用于商业目的。**

在开始之前，请注意以下适配条件。如果不匹配，可能导致运行失败。

| Conditions | Need |
| --- | --- |
| CANN版本 | >=5.0.3 |
| 芯片平台| Ascend310/Ascend310P3 |
| 第三方依赖| 请参考 'requirements.txt' |

## 快速指南

### 1. 拷贝代码

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/recommendation/WideDeep_for_ACL
```

### 2. 生成随机测试数据集

1. 按照连接 [train repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep) 下载outbrain数据集

and preprocess to TFRecords, then move 'outbrain/tfrecords' to './scripts'


2. 生成测试数据集:
```
cd scripts
python3 generate_data.py --path=./outbrain/tfrecords --batchsize=1024
```
在目录 *input_bins/*下生成测试集的bin文件

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型

  [**pb模型下载链接**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/recommendation/WideDeep_for_ACL/widedeep_npu.pb)

  ```
  export batch_size=1024
  atc --model=widedeep_npu.pb --framework=3 --soc_version=Ascend310P3 --output=widedeep_${batch_size}batch --log=error --op_debug_level=3 --input_shape="ad_advertiser:${batch_size},1;ad_id:${batch_size},1;ad_views_log_01scaled:${batch_size},1;doc_ad_category_id:${batch_size},3;doc_ad_days_since_published_log_01scaled:${batch_size},1;doc_ad_entity_id:${batch_size},6;doc_ad_publisher_id:${batch_size},1;doc_ad_source_id:${batch_size},1;doc_ad_topic_id:${batch_size},3;doc_event_category_id:${batch_size},3;doc_event_days_since_published_log_01scaled:${batch_size},1;doc_event_doc_ad_sim_categories_log_01scaled:${batch_size},1;doc_event_doc_ad_sim_entities_log_01scaled:${batch_size},1;doc_event_doc_ad_sim_topics_log_01scaled:${batch_size},1;doc_event_entity_id:${batch_size},6;doc_event_hour_log_01scaled:${batch_size},1;doc_event_id:${batch_size},1;doc_event_publisher_id:${batch_size},1;doc_event_source_id:${batch_size},1;doc_event_topic_id:${batch_size},3;doc_id:${batch_size},1;doc_views_log_01scaled:${batch_size},1;event_country:${batch_size},1;event_country_state:${batch_size},1;event_geo_location:${batch_size},1;event_hour:${batch_size},1;event_platform:${batch_size},1;event_weekend:${batch_size},1;pop_ad_id_conf:${batch_size},1;pop_ad_id_log_01scaled:${batch_size},1;pop_advertiser_id_conf:${batch_size},1;pop_advertiser_id_log_01scaled:${batch_size},1;pop_campain_id_conf_multipl_log_01scaled:${batch_size},1;pop_campain_id_log_01scaled:${batch_size},1;pop_category_id_conf:${batch_size},1;pop_category_id_log_01scaled:${batch_size},1;pop_document_id_conf:${batch_size},1;pop_document_id_log_01scaled:${batch_size},1;pop_entity_id_conf:${batch_size},1;pop_entity_id_log_01scaled:${batch_size},1;pop_publisher_id_conf:${batch_size},1;pop_publisher_id_log_01scaled:${batch_size},1;pop_source_id_conf:${batch_size},1;pop_source_id_log_01scaled:${batch_size},1;pop_topic_id_conf:${batch_size},1;pop_topic_id_log_01scaled:${batch_size},1;traffic_source:${batch_size},1;user_doc_ad_sim_categories_conf:${batch_size},1;user_doc_ad_sim_categories_log_01scaled:${batch_size},1;user_doc_ad_sim_entities_log_01scaled:${batch_size},1;user_doc_ad_sim_topics_conf:${batch_size},1;user_doc_ad_sim_topics_log_01scaled:${batch_size},1;user_has_already_viewed_doc:${batch_size},1;user_views_log_01scaled:${batch_size},1" --out_nodes=eval_predictions:0
  ```

- 编译程序

  ```
  bash build.sh
  ```

- 开始运行:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```

## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model       | **data**  |     Accuracy   |
| :---------------: | :-------: | :-------------: |
| offline Inference | out brain test | 64.9% |

## 参考
[1] https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep
