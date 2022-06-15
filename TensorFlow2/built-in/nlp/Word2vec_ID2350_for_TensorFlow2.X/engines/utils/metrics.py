# -*- coding: utf-8 -*-
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : metrics.py 
# @Software: PyCharm
# import npu_device
# npu_device.open().as_default()
from sklearn import metrics
from config import classifier_config


def cal_metrics(y_true, y_pred):
    """
    指标计算
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    average = classifier_config['metrics_average']
    precision = metrics.precision_score(y_true, y_pred, average=average)
    recall = metrics.recall_score(y_true, y_pred, average=average)
    f1 = metrics.f1_score(y_true, y_pred, average=average)
    each_classes = metrics.classification_report(y_true, y_pred, output_dict=True)
    return {'precision': precision, 'recall': recall, 'f1': f1}, each_classes

