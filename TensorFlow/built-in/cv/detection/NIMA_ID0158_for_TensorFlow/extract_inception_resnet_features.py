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
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.core.protobuf import config_pb2
import tensorflow as tf
from keras import backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from utils.data_loader import train_generator, val_generator

def npu_session_config_init(session_config=None):
    if ((not isinstance(session_config, config_pb2.ConfigProto)) and (not issubclass(type(session_config), config_pb2.ConfigProto))):
        session_config = config_pb2.ConfigProto()
    if (isinstance(session_config, config_pb2.ConfigProto) or issubclass(type(session_config), config_pb2.ConfigProto)):
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    return session_config
sess = tf.Session(config=npu_session_config_init())
K.set_session(sess)
image_size = 224

def _float32_feature_list(floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=floats))
model = InceptionResNetV2(input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
model.summary()
nb_samples = (250000 * 2)
batchsize = 200
with sess.as_default():
    generator = train_generator(batchsize, shuffle=False)
    writer = tf.python_io.TFRecordWriter('weights/inception_resnet_train.tfrecord')
count = 0
for _ in range((nb_samples // batchsize)):
    (x_batch, y_batch) = next(generator)
    with sess.as_default():
        x_batch = model.predict(x_batch, batchsize, verbose=1)
    for (i, (x, y)) in enumerate(zip(x_batch, y_batch)):
        examples = {'features': _float32_feature_list(x.flatten()), 'scores': _float32_feature_list(y.flatten())}
        features = tf.train.Features(feature=examples)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    count += batchsize
    print(('Finished %0.2f percentage storing dataset' % ((count * 100) / float(nb_samples))))
writer.close()
' TRAIN SET '
nb_samples = 5000
batchsize = 200
with sess.as_default():
    generator = val_generator(batchsize)
    writer = tf.python_io.TFRecordWriter('weights/inception_resnet_val.tfrecord')
count = 0
for _ in range((nb_samples // batchsize)):
    (x_batch, y_batch) = next(generator)
    with sess.as_default():
        x_batch = model.predict(x_batch, batchsize, verbose=1)
    for (i, (x, y)) in enumerate(zip(x_batch, y_batch)):
        examples = {'features': _float32_feature_list(x.flatten()), 'scores': _float32_feature_list(y.flatten())}
        features = tf.train.Features(feature=examples)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    count += batchsize
    print(('Finished %0.2f percentage storing dataset' % ((count * 100) / float(nb_samples))))
writer.close()
