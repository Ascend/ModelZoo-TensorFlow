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


from MobileFaceNet_Tensorflow import verification
from scipy.optimize import brentq
from scipy import interpolate
from sklearn import metrics
import tensorflow as tf
import numpy as np
import argparse
import time
import os
import pickle


if __name__ == '__main__':

    om_prediction_path = './om_output/20210915_104452/'
    eval_db_path = './LFW/faces_ms1m_112x112/lfw.bin'
    db_name = 'lfw'
    image_size = [112, 112]
    embedding_size = 128
    test_batch_size = 100
    eval_nrof_folds = 10

    print("start om inference...")
    # loading labels
    _, issame_list = pickle.load(open(eval_db_path, 'rb'), encoding='bytes')
    
    start_time = time.time()
    dataset_len = len(issame_list)*2
    emb_array = np.zeros((dataset_len, embedding_size))
    nrof_batches = dataset_len // test_batch_size
    
    # loading om prediction
    for i in range(len(issame_list) * 2):
        path = om_prediction_path + str(i) + '_output_0.bin'
        emb_array[i] = np.fromfile(path, np.float16)

    duration = time.time() - start_time
    tpr, fpr, accuracy, val, val_std, far = verification.evaluate(emb_array, issame_list, nrof_folds=eval_nrof_folds)

    print("total time %.3f to evaluate %d images of lfw" % (duration,
                                                            dataset_len,))
    print('Accuracy: %1.3f' % (np.mean(accuracy)))
    
    
    
   
    
    