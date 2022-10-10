# encoding:utf-8
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
from npu_bridge.npu_init import *
import sys
import os

import tensorflow as tf

from .cfg import Config
from .other import resize_im
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


sys.path.append(os.getcwd())
from lib.fast_rcnn.config import cfg
from lib.networks.factory import get_network
from lib.fast_rcnn.test import test_ctpn

# from ..lib.networks.factory import get_network
# from ..lib.fast_rcnn.config import cfg
# from..lib.fast_rcnn.test import test_ctpn
'''
load network
输入的名称为'Net_model'
'VGGnet_test'--test
'VGGnet_train'-train
'''


def load_tf_model():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    net = get_network("VGGnet_test")
    # load model
    saver = tf.train.Saver()
    # sess = tf.Session(config=config)
    sess = tf.Session()
    ckpt_path = '/Users/xiaofeng/Code/Github/dataset/CHINESE_OCR/ctpn/ctpn/retrain/ckpt'
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("Tensor_name is : ", key)
        # print(reader.get_tensor(key))
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("load vggnet done")
    return sess, saver, net



# init model
sess, saver, net = load_tf_model()


# 进行文本识别
def ctpn(img):
    """
    text box detect
    """
    scale, max_scale = Config.SCALE, Config.MAX_SCALE
    # 对图像进行resize，输出的图像长宽
    img, f = resize_im(img, scale=scale, max_scale=max_scale)
    scores, boxes = test_ctpn(sess, net, img)
    return scores, boxes, img
