

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

1. Because of this is not a well trained model we test the model with random test dataset

2. Generate random test dataset:
```
cd scripts
mkdir input_bins
python3 generate_random_data.py --path=./input_bins/ --nums=32
```
There will random testdata bin fils under *input_bins/*.

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  [**pb download link**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/recommendation/WideDeep_for_ACL/widedeep_npu.pb)

  ```
  export batch_size=1024
  atc --model=widedeep_npu.pb --framework=3 --soc_version=Ascend310P3 --output=widedeep_${batch_size}_batch --log=error --op_debug_level=3 --input_shape="ad_advertiser_1:${batch_size},1;ad_id_1:${batch_size},1;ad_views_log_01scaled_1:${batch_size},1;doc_ad_category_id_1:${batch_size},3;doc_ad_days_since_published_log_01scaled_1:${batch_size},1;doc_ad_entity_id_1:${batch_size},6;doc_ad_publisher_id_1:${batch_size},1;doc_ad_source_id_1:${batch_size},1;doc_ad_topic_id_1:${batch_size},3;doc_event_category_id_1:${batch_size},3;doc_event_days_since_published_log_01scaled_1:${batch_size},1;doc_event_doc_ad_sim_categories_log_01scaled_1:${batch_size},1;doc_event_doc_ad_sim_entities_log_01scaled_1:${batch_size},1;doc_event_doc_ad_sim_topics_log_01scaled_1:${batch_size},1;doc_event_entity_id_1:${batch_size},6;doc_event_hour_log_01scaled_1:${batch_size},1;doc_event_id_1:${batch_size},1;doc_event_publisher_id_1:${batch_size},1;doc_event_source_id_1:${batch_size},1;doc_event_topic_id_1:${batch_size},3;doc_id_1:${batch_size},1;doc_views_log_01scaled_1:${batch_size},1;event_country_1:${batch_size},1;event_country_state_1:${batch_size},1;event_geo_location_1:${batch_size},1;event_hour_1:${batch_size},1;event_platform_1:${batch_size},1;event_weekend_1:${batch_size},1;pop_ad_id_conf_1:${batch_size},1;pop_ad_id_log_01scaled_1:${batch_size},1;pop_advertiser_id_conf_1:${batch_size},1;pop_advertiser_id_log_01scaled_1:${batch_size},1;pop_campain_id_conf_multipl_log_01scaled_1:${batch_size},1;pop_campain_id_log_01scaled_1:${batch_size},1;pop_category_id_conf_1:${batch_size},1;pop_category_id_log_01scaled_1:${batch_size},1;pop_document_id_conf_1:${batch_size},1;pop_document_id_log_01scaled_1:${batch_size},1;pop_entity_id_conf_1:${batch_size},1;pop_entity_id_log_01scaled_1:${batch_size},1;pop_publisher_id_conf_1:${batch_size},1;pop_publisher_id_log_01scaled_1:${batch_size},1;pop_source_id_conf_1:${batch_size},1;pop_source_id_log_01scaled_1:${batch_size},1;pop_topic_id_conf_1:${batch_size},1;pop_topic_id_log_01scaled_1:${batch_size},1;traffic_source_1:${batch_size},1;user_doc_ad_sim_categories_conf_1:${batch_size},1;user_doc_ad_sim_categories_log_01scaled_1:${batch_size},1;user_doc_ad_sim_entities_log_01scaled_1:${batch_size},1;user_doc_ad_sim_topics_conf_1:${batch_size},1;user_doc_ad_sim_topics_log_01scaled_1:${batch_size},1;user_has_already_viewed_doc_1:${batch_size},1;user_views_log_01scaled_1:${batch_size},1" --out_nodes=head/predictions/probabilities:0
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

|       model       | **data**  |     Mean CosineSimilarity   |
| :---------------: | :-------: | :-------------: |
| offline Inference | random data | 100.0% |

