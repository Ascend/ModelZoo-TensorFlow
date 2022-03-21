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
# =============================================================================
import os
import argparse
import numpy as np
import tensorflow as tf
from os.path import join
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_result", type=str, default="./npu_predict")
    parser.add_argument("--labels", type=str, default="./input_bins/label_ids")
    args = parser.parse_args()

    labels_dir = args.labels
    predict_dir = args.infer_result

    files_list = os.listdir(labels_dir)
    files_list.sort()
    labels = []
    predictions = []

    for i in tqdm(range(len(files_list))):
        file = files_list[i]
        if file.endswith('.bin'):
            index = file.split('.bin')[0]
            label = np.fromfile(join(labels_dir,file), dtype='float32')[0]
            predict = np.fromfile(join(predict_dir,'{}_output_0.bin'.format(index)), dtype='float32')[0]
            labels.append(label)
            predictions.append(predict)

    labels = np.array(labels)
    predictions = np.array(predictions)

    x = tf.placeholder(tf.float32,[len(labels)])
    y = tf.placeholder(tf.float32,[len(predictions)])
    accuracy = tf.contrib.metrics.streaming_pearson_correlation(predictions=y, labels=x)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    _,res = sess.run([accuracy], feed_dict={x:labels, y:predictions})[0]
    print("NPU predict accuracy {:.5f}".format(float(res)))


