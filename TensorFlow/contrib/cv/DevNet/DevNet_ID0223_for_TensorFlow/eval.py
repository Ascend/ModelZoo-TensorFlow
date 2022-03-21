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


import numpy as np
import tensorflow as tf
import argparse
import time
import os

from scipy.sparse import vstack, csc_matrix
from utils import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file
from sklearn.model_selection import train_test_split
from devnet import *


tf.set_random_seed(42)
np.random.seed(42)
MAX_INT = np.iinfo(np.int32).max
data_format = 0


def load_model_weight_predict(model_name, network_depth, x_test):
    """
    load the saved weights to make predictions
    """
    scoring_network = DeviationNetwork(x_test.shape, network_depth, learning_rate=0.001)
    scoring_network.load_weights(model_name)

    if data_format == 0:
        scores = scoring_network.predict(x_test)
    else:
        data_size = x_test.shape[0]
        scores = np.zeros([data_size, 1])
        count = 512
        i = 0
        while i < data_size:
            subset = x_test[i:count].toarray()
            scores[i:count] = scoring_network.predict(subset)
            if i % 1024 == 0:
                print(i)
            i = count
            count += 512
            if count > data_size:
                count = data_size
        assert count == data_size
    return scores


def eval_devnet(args):
    """
    eval devnet
    """
    names = ['annthyroid_21feat_normalised']
    network_depth = int(args.network_depth)
    for nm in names:
        filename = nm.strip()
        x, labels = dataLoading(args.input_path + filename + ".csv")
        outlier_indices = np.where(labels == 1)[0]
        outliers = x[outlier_indices]
        n_outliers_org = outliers.shape[0]

        train_time = 0
        test_time = 0
        x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42,
                                                            stratify=labels)
        y_test = np.array(y_test)
        x_test = x_test.astype(np.float32)
        model_name = args.model_path
        scores = load_model_weight_predict(model_name, network_depth, x_test)
        rauc, ap = aucPerformance(scores, y_test)
        x_test.tofile("test_input.bin")
        y_test.tofile("test_label.bin")

parser = argparse.ArgumentParser()
parser.add_argument("--network_depth", choices=['1', '2', '4'], default='4',
                    help="the depth of the network architecture")
parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
parser.add_argument("--model_path", type=str, default='./model/devnet_annthyroid_21feat_normalised_0.02cr_512bs_30ko_4d', help="the path of the model ckpt")
args = parser.parse_args()


args = parser.parse_args()
eval_devnet(args)
