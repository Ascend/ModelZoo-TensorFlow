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

from sklearn.metrics import log_loss, roc_auc_score
import numpy as np

def np_sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    _log_loss = log_loss(labels, mse)
    print("AUC-ROC: %.4f, LOG_LOSS: %.4f" % (roc_auc, _log_loss))
    return roc_auc, _log_loss;

def eval_om(label_bin_file_path, om_output_path):
    test_label = []
    score = []
    eps = 1e-6
    for i in range(2049):
        if i == 999:
            continue
        label_bin_file = label_bin_file_path + "/" + str(i) + ".bin"
        om_output = om_output_path + "/" + str(i) + "_output_0.bin"
        test_label.extend(list(np.fromfile(label_bin_file, dtype=np.int32)))
        score.extend(list(np.fromfile(om_output, dtype=np.float32)))
    test_label = np.array(test_label)
    score = np.array(score)
    score = np_sigmoid(score)

    _min_ = len(np.where(score < eps)[0])
    _max_ = len(np.where(score > 1-eps)[0])
    print('%d samples are evaluated' % len(test_label))
    if _min_ + _max_ > 0:
        print('EPS: %g, %d (%.2f) < eps, %d (%.2f) > 1-eps, %d (%.2f) are truncated' %
                (eps, _min_, _min_ / len(score), _max_, _max_ / len(score), _min_ + _max_,
                (_min_ + _max_) / len(score)))
    score[score < eps] = eps
    score[score > 1-eps] = 1-eps

    aucPerformance(score, test_label)

eval_om("../data/labels", "../om_output/20210708_221703")