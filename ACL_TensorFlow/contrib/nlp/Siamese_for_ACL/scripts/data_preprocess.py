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
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("eval_filepath", "./dataset/validation.txt0", "Evaluate on this data")
tf.flags.DEFINE_string("vocab_filepath", "./dataset/vocab", "Load training time vocabulary")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()

if __name__ == "__main__":
    os.makedirs("./input_x1", exist_ok=True)
    os.makedirs("./input_x2", exist_ok=True)
    os.makedirs("ground_truth", exist_ok=True)

    # load data and map id-transform based on training time vocabulary
    inpH = InputHelper()
    x1_test, x2_test, y_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.vocab_filepath, 30)
    # Generate batches for one epoch
    batches = inpH.batch_iter(list(zip(x1_test, x2_test, y_test)), 2 * FLAGS.batch_size, 1, shuffle=False)
    index = 0
    for db in batches:
        x1_dev_b, x2_dev_b, y_dev_b = zip(*db)
        np_x1 = np.array(x1_dev_b, dtype='int32')
        np_x2 = np.array(x2_dev_b, dtype='int32')
        np_y = np.array(y_dev_b, dtype='int32')

        if np_x1.shape[0] == 2 * FLAGS.batch_size:
            np_x1.tofile("input_x1/{}.bin".format(str(index).zfill(6)))
            np_x2.tofile("input_x2/{}.bin".format(str(index).zfill(6)))
            np_y.tofile("ground_truth/{}.bin".format(str(index).zfill(6)))
            index += 1
    print("Preprocess dataset finished!")