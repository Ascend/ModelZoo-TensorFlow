English|[中文](README.md)

# ESMM Inference for Tensorflow 

This repository provides a script and recipe to Inference of the ESMM model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/recommendation/ESMM_for_ACL
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

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  [**pb download link**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/recommendation/ESMM_for_ACL/esmm.pb)

  ```
  export batch_size=1
  atc --model=./esmm.pb --framework=3 --output=./esmm_${batch_size}batch_input_int64 --soc_version=Ascend310P3 --input_shape="userindex:${batch_size},1;pvc_id:${batch_size},1;city_id:${batch_size},1;level:${batch_size},1;gender:${batch_size},1;age:${batch_size},1;predict_gender:${batch_size},1;predict_age:${batch_size},1;job_id:${batch_size},1;style_ids_id/values:${batch_size};style_ids_id/indices:${batch_size},2;lang_ids_id/values:${batch_size};lang_ids_id/indices:${batch_size},2;artist_ids_id/values:${batch_size};artist_ids_id/indices:${batch_size},2;c_artist_ids:${batch_size},1;c_category_ctr_week:${batch_size},1;c_category_ctr_month:${batch_size},1;c_user_ctr_week:${batch_size},1;c_user_ctr_month:${batch_size},1;user_ctr_week:${batch_size},1;user_ctr_month:${batch_size},1;mlog_index:${batch_size},1;s_sourceid:${batch_size},1;songid_index:${batch_size},1;theme_fst_tags/values:${batch_size};theme_fst_tags/indices:${batch_size},2;theme_sed_tags/values:${batch_size};theme_sed_tags/indices:${batch_size},2;contentdesctags/values:${batch_size};contentdesctags/indices:${batch_size},2;qualitytags/values:${batch_size};qualitytags/indices:${batch_size},2;timenodetags/values:${batch_size};timenodetags/indices:${batch_size},2;publish_time:${batch_size},1;artistid:${batch_size},1;artists/values:${batch_size};artists/indices:${batch_size},2;square_impress_7d:${batch_size},1;square_click_7d:${batch_size},1;square_ctr_7d:${batch_size},1;c_gender_ctr:${batch_size},1;impress_1d:${batch_size},1;view_1d:${batch_size},1;zan_1d:${batch_size},1;ctr_1d:${batch_size},1;ztr_1d:${batch_size},1;avg_time_1d:${batch_size},1;complete_ctr_1d:${batch_size},1;impress_3d:${batch_size},1;view_3d:${batch_size},1;zan_3d:${batch_size},1;ctr_3d:${batch_size},1;ztr_3d:${batch_size},1;avg_time_3d:${batch_size},1;complete_ctr_3d:${batch_size},1;impress_7d:${batch_size},1;view_7d:${batch_size},1;zan_7d:${batch_size},1;ctr_7d:${batch_size},1;ztr_7d:${batch_size},1;avg_time_7d:${batch_size},1;complete_ctr_7d:${batch_size},1;impress_14d:${batch_size},1;view_14d:${batch_size},1;zan_14d:${batch_size},1;ctr_14d:${batch_size},1;ztr_14d:${batch_size},1;avg_time_14d:${batch_size},1;complete_ctr_14d:${batch_size},1;s_impress_3h:${batch_size},1;s_validView_3h:${batch_size},1;s_ctr_3h:${batch_size},1;s_user_play_list:${batch_size},10;s_user_zan_list:${batch_size},10;s_weekday:${batch_size},1;s_hour:${batch_size},1;s_user_play30_list:${batch_size},20;s_user_play5_list:${batch_size},20;s_user_play60_list:${batch_size},20;s_user_impress_list:${batch_size},20;s_user_search_list:${batch_size},20;s_user_song_list:${batch_size},20" --log info
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

