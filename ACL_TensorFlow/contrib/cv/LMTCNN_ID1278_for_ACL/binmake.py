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

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
from data import multiinputs
import os
import json


tf.app.flags.DEFINE_string('data_dir','./tfrecord/train_val_test_per_fold_agegender/test_fold_is_4',
	'valiation and testing set directory')
FLAGS = tf.app.flags.FLAGS

tfpath = os.path.join(FLAGS.data_dir, 'test.tfrecord')

if __name__ == '__main__':
    if not os.path.exists('imagebin'):
        os.mkdir('imagebin') 
    age_input_file = os.path.join(FLAGS.data_dir, 'mdage.json')
    gender_input_file = os.path.join(FLAGS.data_dir, 'mdgender.json')
    with open(age_input_file, 'r') as fage, open(gender_input_file, 'r') as fgender:
        mdage = json.load(fage)
        mdgender = json.load(fgender)
    num_eval = mdage['test_counts']
    dataset =  multiinputs(data_dir = tfpath, batch_size=1 ,train=False,num_epochs= 1)
    iterator = dataset.make_one_shot_iterator()
    images0, agelabels0, genderlabels0 = iterator.get_next()
    with tf.Session() as sess:
        agelabel = np.ones(1)
        genderlabel = np.ones(1)
        for i in range(0,num_eval):
            images, agelabels_1, genderlabels_1 = sess.run([images0, agelabels0, genderlabels0])
            agelabels_1 = (np.reshape(agelabels_1, (1))).astype(np.int32)
            genderlabels_1 = (np.reshape(genderlabels_1, (1))).astype(np.int32)

            if i ==0:
                agelabel  = agelabels_1
                genderlabel = genderlabels_1
            else:
                agelabel =np.vstack((agelabel,agelabels_1))
                genderlabel =np.vstack((genderlabel,genderlabels_1))
            name = str(i).zfill(4)
            images.astype(np.float32).tofile(f"imagebin/{name}.bin")
    if not os.path.exists('label'):
        os.mkdir('label')
    np.save('label/agelabel_1.npy',agelabel)
    np.save('label/genderlabel_1.npy',genderlabel)