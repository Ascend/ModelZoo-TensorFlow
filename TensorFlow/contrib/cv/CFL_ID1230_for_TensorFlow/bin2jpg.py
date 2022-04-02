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
from numpy import *
import os
from config import *
import glob
from PIL import Image
import tensorflow as tf
import imageio


with tf.Session() as sess:
    prediction=np.fromfile("tf_cfl_output_0.bin",dtype=float32).reshape([1,64,128,2])
    pred_edges, pred_corners = tf.split(prediction, [1, 1], 3)

    emap,cmap = sess.run([tf.nn.sigmoid(pred_edges),tf.nn.sigmoid(pred_corners)])

    imageio.imwrite("p1_EM_test.jpg",emap[0,:,:,0])
    imageio.imwrite("p1_CM_test.jpg",cmap[0,:,:,0])
