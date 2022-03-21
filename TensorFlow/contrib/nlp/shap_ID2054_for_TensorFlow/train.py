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

from npu_bridge.npu_init import *
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
import json
import shap
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten

import os


import tensorflow as tf
from npu_bridge.npu_init import *

sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["dynamic_input"].b = True
custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)

# config_proto = tf.ConfigProto()
# custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
# custom_op.name = 'NpuOptimizer'
# # 开启profiling采集
# custom_op.parameter_map["profiling_mode"].b = True
# # 仅采集任务轨迹数据
# # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/cache/profiling","task_trace":"on"}')
# # 采集任务轨迹数据和迭代轨迹数据。可先仅采集任务轨迹数据，如果仍然无法分析到具体问题，可再采集迭代轨迹数据
# custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/cache/profiling","task_trace":"on","fp_point":"","bp_point":""}')
# npu_keras_sess = set_keras_session_npu_config(config=config_proto)


# config = tf.ConfigProto()
# custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
# custom_op.name = "NpuOptimizer"
# custom_op.parameter_map["use_off_line"].b = True
# custom_op.parameter_map["profiling_mode"].b = True
# # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/ma-user/modelarts/user-job-dir/code/profiling","task_trace":"on","aicpu":"on","fp_point":"","bp_point":""}')
# custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
#     '{"output":"/cache/profiling","task_trace":"on","aicpu":"on","fp_point":"","bp_point":""}')
# # config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关


# session_config = tf.ConfigProto(allow_soft_placement=True)
# custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
# custom_op.name = 'NpuOptimizer'
# # 开启profiling采集
# custom_op.parameter_map["profiling_mode"].b = True
# # 仅采集任务轨迹数据
# custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/HwHiAiUser/output","task_trace":"on"}')
# # 采集任务轨迹数据和迭代轨迹数据。可先仅采集任务轨迹数据，如果仍然无法分析到具体问题，可再采集迭代轨迹数据
# # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/HwHiAiUser/output","task_trace":"on","training_trace":"on","fp_point":"resnet_model/conv2d/Conv2Dresnet_model/batch_normalization/FusedBatchNormV3_Reduce","bp_point":"gradients/AddN_70"}')




def VGG16():
    """
    Return a vgg16 model.

    Keras has a built in vgg16 model which omits Dropouts.
    I don't want to omit the dropouts as they are part of
    the original vgg16 model. therefore I have to define the
    vgg16 model myself.
    """
    img_input = Input(shape=(224, 224, 3), name='input')

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5, name='Dropout1')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='Dropout2')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    vgg16 = Model(inputs=img_input, outputs=x)

    return vgg16


model = VGG16()
model.load_weights('VGG16/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

X,y = shap.datasets.imagenet50()
to_explain = X[[39,41]]

# load the ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)

# explain how the input to the 7th layer of the model explains the top two classes
def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
    return K.get_session().run(model.layers[layer].input, feed_dict)
e = shap.GradientExplainer(
    (model.layers[7].input, model.layers[-1].output),
    map2layer(X, 7),
    local_smoothing=0 # std dev of smoothing noise
)
shap_values,indexes = e.shap_values(map2layer(to_explain, 7), ranked_outputs=2)

# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# plot the explanations
shap.image_plot(shap_values, to_explain, index_names)

sess.close()