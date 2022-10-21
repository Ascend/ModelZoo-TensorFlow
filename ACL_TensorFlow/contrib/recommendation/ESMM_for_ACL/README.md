中文|[English](README_EN.md)

# ESMM TensorFlow离线推理

此链接提供ESMM TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/recommendation/ESMM_for_ACL
```

### 2. 生成随机测试数据集

1. 由于这不是一个经过良好训练的模型，我们使用随机测试数据集测试模型
2. 生成随机测试数据集:
```
cd scripts
mkdir input_bins
python3 generate_random_data.py --path=./input_bins/ --nums=32
```
在*input_bins/*下会有随机的testdata bin文件。

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型

  [**pb模型下载链接**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/recommendation/ESMM_for_ACL/esmm.pb)

  ```
  export batch_size=1
  atc --model=./esmm.pb --framework=3 --output=./esmm_${batch_size}batch_input_int64 --soc_version=Ascend310P3 --input_shape="userindex:${batch_size},1;pvc_id:${batch_size},1;city_id:${batch_size},1;level:${batch_size},1;gender:${batch_size},1;age:${batch_size},1;predict_gender:${batch_size},1;predict_age:${batch_size},1;job_id:${batch_size},1;style_ids_id/values:${batch_size};style_ids_id/indices:${batch_size},2;lang_ids_id/values:${batch_size};lang_ids_id/indices:${batch_size},2;artist_ids_id/values:${batch_size};artist_ids_id/indices:${batch_size},2;c_artist_ids:${batch_size},1;c_category_ctr_week:${batch_size},1;c_category_ctr_month:${batch_size},1;c_user_ctr_week:${batch_size},1;c_user_ctr_month:${batch_size},1;user_ctr_week:${batch_size},1;user_ctr_month:${batch_size},1;mlog_index:${batch_size},1;s_sourceid:${batch_size},1;songid_index:${batch_size},1;theme_fst_tags/values:${batch_size};theme_fst_tags/indices:${batch_size},2;theme_sed_tags/values:${batch_size};theme_sed_tags/indices:${batch_size},2;contentdesctags/values:${batch_size};contentdesctags/indices:${batch_size},2;qualitytags/values:${batch_size};qualitytags/indices:${batch_size},2;timenodetags/values:${batch_size};timenodetags/indices:${batch_size},2;publish_time:${batch_size},1;artistid:${batch_size},1;artists/values:${batch_size};artists/indices:${batch_size},2;square_impress_7d:${batch_size},1;square_click_7d:${batch_size},1;square_ctr_7d:${batch_size},1;c_gender_ctr:${batch_size},1;impress_1d:${batch_size},1;view_1d:${batch_size},1;zan_1d:${batch_size},1;ctr_1d:${batch_size},1;ztr_1d:${batch_size},1;avg_time_1d:${batch_size},1;complete_ctr_1d:${batch_size},1;impress_3d:${batch_size},1;view_3d:${batch_size},1;zan_3d:${batch_size},1;ctr_3d:${batch_size},1;ztr_3d:${batch_size},1;avg_time_3d:${batch_size},1;complete_ctr_3d:${batch_size},1;impress_7d:${batch_size},1;view_7d:${batch_size},1;zan_7d:${batch_size},1;ctr_7d:${batch_size},1;ztr_7d:${batch_size},1;avg_time_7d:${batch_size},1;complete_ctr_7d:${batch_size},1;impress_14d:${batch_size},1;view_14d:${batch_size},1;zan_14d:${batch_size},1;ctr_14d:${batch_size},1;ztr_14d:${batch_size},1;avg_time_14d:${batch_size},1;complete_ctr_14d:${batch_size},1;s_impress_3h:${batch_size},1;s_validView_3h:${batch_size},1;s_ctr_3h:${batch_size},1;s_user_play_list:${batch_size},10;s_user_zan_list:${batch_size},10;s_weekday:${batch_size},1;s_hour:${batch_size},1;s_user_play30_list:${batch_size},20;s_user_play5_list:${batch_size},20;s_user_play60_list:${batch_size},20;s_user_impress_list:${batch_size},20;s_user_search_list:${batch_size},20;s_user_song_list:${batch_size},20" --log info
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
|       model       | **data**  |     Mean CosineSimilarity   |
| :---------------: | :-------: | :-------------: |
| offline Inference | random data | 100.0% |

