-   [������Ϣ]
-   [����]
-   [ѵ������׼��]
-   [ѵ������]
-   [����ָ��]

<h2 id="������Ϣ.md">������Ϣ</h2>

**�����ߣ�Publisher����Huawei**

**Ӧ������Application Domain����?Missing pattern**

**�汾��Version����1.1**

**�޸�ʱ�䣨Modified�� ��2021.12.23**

**��С��Size����389K**

**��ܣ�Framework����TensorFlow 1.15.0**

**ģ�͸�ʽ��Model Format����ckpt**

**���ȣ�Precision����Mixed**

**��������Processor�����N��910**

**Ӧ�ü���Categories����Official**

**������Description��������TensorFlow��ܵĳ����߼��ѵ������** 

<h2 id="����.md">����</h2>

���㷨��һ�ֿ��ٳ��������㷨���ܹ�������������ͳ����任�����㷨�������������תΪʵ���ָ����⣬�Ӷ�ÿ�������߸����γ�һ��ʵ����ʵ��ʵ�ֶ˵���ѵ�����ڷָ����������ϳ���֮ǰ������һ����ѧϰ�õ�͸�ӱ任����ͼ���������ֵ�������̶������ͼ���Ա���ȷ���ڵ�·ƽ��仯�µĳ�������ϵ�³���ԡ����㷨��tuSimple���ݼ�����֤����ȡ�ý������ƵĽ����

- �ο����ģ�

[D. Neven, B. D. Brabandere, S. Georgoulis, M. Proesmans and L. V. Gool, "Towards End-to-End Lane Detection: an Instance Segmentation Approach," 2018 IEEE Intelligent Vehicles Symposium (IV), 2018, pp. 286-291, doi: 10.1109/IVS.2018.8500547.] 

- �ο�ʵ�֣�

    

- ����N�� AI ��������ʵ�֣�
  
  [https://gitee.com/jingzhongrenxc/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/MNN-LANENET_ID1251_for_TensorFlow](https://gitee.com/jingzhongrenxc/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/MNN-LANENET_ID1251_for_TensorFlow)      


- ͨ��Git��ȡ��Ӧcommit\_id�Ĵ��뷽�����£�
  
    ```
    git clone {repository_url}    # ��¡�ֿ�Ĵ���
    cd {repository_name}    # �л���ģ�͵Ĵ����Ŀ¼
    git checkout  {branch}    # �л�����Ӧ��֧
    git reset --hard ��commit_id��     # �������õ���Ӧ��commit_id
    cd ��code_path��    # �л���ģ�ʹ�������·�������ֿ���ֻ�и�ģ�ͣ��������л�
    ```

## Ĭ������<a name="section91661242121611"></a>

- ѵ�����ݼ���tuSimple���ݼ�������Ϊ�û��ο�ʾ������
  -���ݼ���ȡ��ַ��https://link.zhihu.com/?target=https%3A//github.com/TuSimple/tusimple-benchmark/issues/3

- ѵ������

    - SNAPSHOT_EPOCH: 8
    - BATCH_SIZE: 16
    - VAL_BATCH_SIZE: 16
    - EPOCH_NUMS: 905
    - WARM_UP:
        - ENABLE: True
        - EPOCH_NUMS: 8
    - LR: 0.001
    - LR_POLYNOMIAL_POWER: 0.9
    - MOMENTUM: 0.9
    - WEIGHT_DECAY: 0.0005
    - MOVING_AVE_DECAY: 0.9995

## ֧������<a name="section1899153513554"></a>

| �����б�  | �Ƿ�֧�� |
|-------    |------    |
| ��Ͼ���  |  ��      |


## ��Ͼ���ѵ��<a name="section168064817164"></a>

�N��910 AI�������ṩ�Զ���Ͼ��ȹ��ܣ��������ȫ����float32�������͵����ӣ��������õ��Ż����ԣ��Զ�������float32�����ӽ��;��ȵ�float16���Ӷ��ھ�����ʧ��С�����������ϵͳ���ܲ������ڴ�ʹ�á�

## ������Ͼ���<a name="section20779114113713"></a>

�ű���Ĭ�Ϲرջ�Ͼ��ȣ���Ϊ����֮��������������

  ```custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")



     100%|��������������������| 192/192 [02:00<00:00,  1.59it/s]
  
  ```


<h2 id="ѵ������׼��.md">ѵ������׼��</h2>

Ӳ��������Ascend: 1*Ascend 910(32GB) | ARM: 24 �� 96GB
���л�����ascend-share/5.1.rc2.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_0602

    
  
<h2 id="ѵ������.md">ѵ������</h2>

- ���ݼ�׼����

  ���ݼ�Ҫ�����£�

 ����� data/training_data_example �ļ��нṹ�� ������Ҫ����һ��train.txt��һ��val.txt����¼����ѵ��ģ�͵����ݡ�
ѵ����������������ɣ������Ʒָ��ǩ�ļ���ʵ���ָ��ǩ�ļ���ԭʼͼ�� �����Ʒָ�ʹ�� 255 ��ʾ�����ֶΣ����ಿ��ʹ�� 0�� ��ʵ��ʹ�ò�ͬ������ֵ����ʾ��ͬ�ĳ������������Ϊ 0�����е�ѵ��ͼ�񽫸��������ļ����ŵ���ͬ�ı�����
ʹ���½ű����� tensorflow ��¼�ļ���
python tools/make_tusimple_tfrecords.py 

-�ű���ʾ������
������ config                              
    ������ tusimple_lanenet.yaml          //�����ļ�
������ data_provider          //׼������
    ������ lanenet_data_feed_pipline.py 
    ������ tf_io_pipline_tools.py
������ lanenet_model          //����ģ��
    ������ lanenet.py
    ������ lanenet_back_end.py
    ������ lanenet_discriminative_loss.py
    ������ lanenet_front_end.py
    ������ lanenet_postprocess.py
������ local_utils                                   
    ������ config_utils
        ������ parse_config_utils.py
    ������ log_util
        ������ init_logger.py
������ mnn_project 
    ������ freeze_lanenet_model.py
������ semantic_segmentation_zoo
    ������ bisenet_v2.py
    ������ cnn_basenet.py
    ������ vgg16_based_fcn.py
������ tools
    ������ evaluate_lanenet_on_tusimple.py
    ������ evaluate_model_utils.py
    ������ generate_tusimple_dataset.py
    ������ make_tusimple_tfrecords.py
    ������ test_lanenet.py
    ������ train_lanenet_tusimple.py          //���
������ trainner 
    ������ tusimple_lanenet_multi_gpu_trainner.py
    ������ tusimple_lanenet_single_gpu_trainner.py

 
- ģ��ѵ����

   ʹ��pycharm��ModelArtsѵ�������ļ�Ϊ��/tools/train_lanenet_tusimple.py




## ѵ������<a name="section1589455252218"></a>

��������ѵ��

1.  ѵ���ű�log�а���������Ϣ��

```
  0%|          | 0/192 [00:00<?, ?it/s]2022-06-16 22:11:40.757045: I tf_adapter/kernels/geop_npu.cc:817] The model has been compiled on the Ascend AI processor, current graph id is: 11

train loss: 34.08809, b_loss: 1.56111, i_loss: 28.89311:   0%|          | 0/192 [03:17<?, ?it/s]
train loss: 34.08809, b_loss: 1.56111, i_loss: 28.89311:   1%|          | 1/192 [03:17<10:28:28, 197.42s/it]
train loss: 33.28113, b_loss: 1.53935, i_loss: 28.10791:   1%|          | 1/192 [03:18<10:28:28, 197.42s/it]
train loss: 33.28113, b_loss: 1.53935, i_loss: 28.10791:   1%|          | 2/192 [03:18<7:18:14, 138.39s/it] 
train loss: 32.75388, b_loss: 1.61211, i_loss: 27.50790:   1%|          | 2/192 [03:18<7:18:14, 138.39s/it]
train loss: 32.75388, b_loss: 1.61211, i_loss: 27.50790:   2%|��         | 3/192 [03:18<5:05:44, 97.06s/it] 
train loss: 33.34206, b_loss: 1.55278, i_loss: 28.15541:   2%|��         | 3/192 [03:19<5:05:44, 97.06s/it]
train loss: 33.34206, b_loss: 1.55278, i_loss: 28.15541:   2%|��         | 4/192 [03:19<3:33:27, 68.13s/it]
train loss: 34.02236, b_loss: 1.59200, i_loss: 28.79649:   2%|��         | 4/192 [03:19<3:33:27, 68.13s/it]
train loss: 34.02236, b_loss: 1.59200, i_loss: 28.79649:   3%|��         | 5/192 [03:19<2:29:12, 47.88s/it]
train loss: 33.55459, b_loss: 1.55814, i_loss: 28.36258:   3%|��         | 5/192 [03:20<2:29:12, 47.88s/it]
train loss: 33.55459, b_loss: 1.55814, i_loss: 28.36258:   3%|��         | 6/192 [03:20<1:44:28, 33.70s/it]
train loss: 34.67654, b_loss: 1.56015, i_loss: 29.48252:   3%|��         | 6/192 [03:21<1:44:28, 33.70s/it]
train loss: 34.67654, b_loss: 1.56015, i_loss: 29.48252:   4%|��         | 7/192 [03:21<1:13:19, 23.78s/it]
train loss: 33.96736, b_loss: 1.56802, i_loss: 28.76547:   4%|��         | 7/192 [03:21<1:13:19, 23.78s/it]
train loss: 33.96736, b_loss: 1.56802, i_loss: 28.76547:   4%|��         | 8/192 [03:21<51:36, 16.83s/it]  
train loss: 33.22740, b_loss: 1.53942, i_loss: 28.05412:   4%|��         | 8/192 [03:22<51:36, 16.83s/it]
train loss: 33.22740, b_loss: 1.53942, i_loss: 28.05412:   5%|��         | 9/192 [03:22<36:30, 11.97s/it]
train loss: 35.09825, b_loss: 1.53474, i_loss: 29.92964:   5%|��         | 9/192 [03:23<36:30, 11.97s/it]
train loss: 35.09825, b_loss: 1.53474, i_loss: 29.92964:   5%|��         | 10/192 [03:23<25:58,  8.56s/it]
train loss: 34.73063, b_loss: 1.53588, i_loss: 29.56088:   5%|��         | 10/192 [03:23<25:58,  8.56s/it]
train loss: 34.73063, b_loss: 1.53588, i_loss: 29.56088:   6%|��         | 11/192 [03:23<18:38,  6.18s/it]
train loss: 33.60005, b_loss: 1.60795, i_loss: 28.35823:   6%|��         | 11/192 [03:24<18:38,  6.18s/it]
train loss: 33.60005, b_loss: 1.60795, i_loss: 28.35823:   6%|��         | 12/192 [03:24<13:31,  4.51s/it]
train loss: 32.77514, b_loss: 1.59790, i_loss: 27.54338:   6%|��         | 12/192 [03:24<13:31,  4.51s/it]
train loss: 32.77514, b_loss: 1.59790, i_loss: 27.54338:   7%|��         | 13/192 [03:24<09:58,  3.35s/it]
train loss: 33.52126, b_loss: 1.57790, i_loss: 28.30949:   7%|��         | 13/192 [03:25<09:58,  3.35s/it]
train loss: 33.52126, b_loss: 1.57790, i_loss: 28.30949:   7%|��         | 14/192 [03:25<07:30,  2.53s/it]
train loss: 32.66967, b_loss: 1.52915, i_loss: 27.50666:   7%|��         | 14/192 [03:26<07:30,  2.53s/it]
train loss: 32.66967, b_loss: 1.52915, i_loss: 27.50666:   8%|��         | 15/192 [03:26<05:45,  1.95s/it]
train loss: 34.08065, b_loss: 1.56651, i_loss: 28.88028:   8%|��         | 15/192 [03:26<05:45,  1.95s/it]
train loss: 34.08065, b_loss: 1.56651, i_loss: 28.88028:   8%|��         | 16/192 [03:26<04:34,  1.56s/it]
...
train loss: 27.36981, b_loss: 1.53718, i_loss: 22.19878: 100%|��������������������| 192/192 [05:20<00:00,  1.67s/it]
2022-06-16 22:16:42.990 | INFO     | trainner.tusimple_lanenet_single_gpu_trainner:train:366 - => Epoch: 1 Time: 2022-06-16 22:16:42 Train loss: 31.00657 ...

<h2 id="����ָ��.md">����ָ��</h2>
ѵ����Loss

| gpu   | npu  |ԭ���� |
|--------|------|-----------|
|   1.9305   |  0.991   |   ԭ��ѵ����������(40k��)��ֻ����905��   | 
