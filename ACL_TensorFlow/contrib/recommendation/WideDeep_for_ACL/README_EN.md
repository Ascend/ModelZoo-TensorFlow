English|[中文](README.md)

# Wide&Deep Inference for Tensorflow 

This repository provides a script and recipe to Inference of the Wide&Deep model.

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310P3 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/recommendation/WideDeep_for_ACL
```

### 2. Generate random test dataset

1. Follow the [train repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep) guide download outbrain dataset

and preprocess to TFRecords, then move 'outbrain/tfrecords' to './scripts'


2. Generate test dataset:
```
cd scripts
python3 generate_data.py --path=./outbrain/tfrecords --batchsize=1024
```
There will generate testdata bin fils under *input_bins/*.

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  [**pb download link**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/recommendation/WideDeep_for_ACL/widedeep_npu.pb)

  ```
  export batch_size=1024
  atc --model=widedeep_npu.pb --framework=3 --soc_version=Ascend310P3 --output=widedeep_${batch_size}batch --log=error --op_debug_level=3 --input_shape="ad_advertiser:${batch_size},1;ad_id:${batch_size},1;ad_views_log_01scaled:${batch_size},1;doc_ad_category_id:${batch_size},3;doc_ad_days_since_published_log_01scaled:${batch_size},1;doc_ad_entity_id:${batch_size},6;doc_ad_publisher_id:${batch_size},1;doc_ad_source_id:${batch_size},1;doc_ad_topic_id:${batch_size},3;doc_event_category_id:${batch_size},3;doc_event_days_since_published_log_01scaled:${batch_size},1;doc_event_doc_ad_sim_categories_log_01scaled:${batch_size},1;doc_event_doc_ad_sim_entities_log_01scaled:${batch_size},1;doc_event_doc_ad_sim_topics_log_01scaled:${batch_size},1;doc_event_entity_id:${batch_size},6;doc_event_hour_log_01scaled:${batch_size},1;doc_event_id:${batch_size},1;doc_event_publisher_id:${batch_size},1;doc_event_source_id:${batch_size},1;doc_event_topic_id:${batch_size},3;doc_id:${batch_size},1;doc_views_log_01scaled:${batch_size},1;event_country:${batch_size},1;event_country_state:${batch_size},1;event_geo_location:${batch_size},1;event_hour:${batch_size},1;event_platform:${batch_size},1;event_weekend:${batch_size},1;pop_ad_id_conf:${batch_size},1;pop_ad_id_log_01scaled:${batch_size},1;pop_advertiser_id_conf:${batch_size},1;pop_advertiser_id_log_01scaled:${batch_size},1;pop_campain_id_conf_multipl_log_01scaled:${batch_size},1;pop_campain_id_log_01scaled:${batch_size},1;pop_category_id_conf:${batch_size},1;pop_category_id_log_01scaled:${batch_size},1;pop_document_id_conf:${batch_size},1;pop_document_id_log_01scaled:${batch_size},1;pop_entity_id_conf:${batch_size},1;pop_entity_id_log_01scaled:${batch_size},1;pop_publisher_id_conf:${batch_size},1;pop_publisher_id_log_01scaled:${batch_size},1;pop_source_id_conf:${batch_size},1;pop_source_id_log_01scaled:${batch_size},1;pop_topic_id_conf:${batch_size},1;pop_topic_id_log_01scaled:${batch_size},1;traffic_source:${batch_size},1;user_doc_ad_sim_categories_conf:${batch_size},1;user_doc_ad_sim_categories_log_01scaled:${batch_size},1;user_doc_ad_sim_entities_log_01scaled:${batch_size},1;user_doc_ad_sim_topics_conf:${batch_size},1;user_doc_ad_sim_topics_log_01scaled:${batch_size},1;user_has_already_viewed_doc:${batch_size},1;user_views_log_01scaled:${batch_size},1" --out_nodes=eval_predictions:0
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```

## Performance

### Result

Our result was obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**  |     Accuracy   |
| :---------------: | :-------: | :-------------: |
| offline Inference | out brain test | 64.9% |

## Reference
[1] https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep
