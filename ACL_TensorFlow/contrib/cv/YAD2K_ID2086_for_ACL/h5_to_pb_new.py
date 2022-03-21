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

import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class SpaceToDepth(layers.Layer):
    def __init__(self, block_size=2, **kwargs):
        self.block_size = 2
        super(SpaceToDepth, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        batch, height, width, depth = K.int_shape(x)
        batch = -1
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        y = K.reshape(x, (batch, reduced_height, self.block_size,
                          reduced_width, self.block_size, depth))
        z = K.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
        t = K.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size ** 2))
        return t

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0], input_shape[1] // self.block_size, input_shape[2] // self.block_size,
                 input_shape[3] * self.block_size ** 2)
        return shape


def yolo_choose(y_true, y_pre):
    return y_pre


def yolo_head(feats, anchors, clsnum):
    """feats 网络输出（batch,13,13,125）
       anchors:(5,2)
       clsnum:20
    """
    feats = tf.convert_to_tensor(feats, dtype=tf.float32)  # 将numpy数组转换为tf张量，
    num_anchors = len(anchors)  # 5
    # 转化为：shape:（ batch, height, width, 5, 2）
    anchors_tensor = K.reshape(tf.convert_to_tensor(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])

    conv_dims = K.shape(feats)[1:3]  # 得到特征图的第1,2个维度---> [13, 13] ,shape：（2，）
    conv_height_index = K.arange(0, stop=conv_dims[0])  # 得到[0--12]的一维数组，代表高
    conv_width_index = K.arange(0, stop=conv_dims[1])  # 得到[0--12]的一维数组，代表宽

    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])  # 进行一维平铺，[0--12 0--12] 重复13次 shape(13*13，)
    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0),
                              [conv_dims[0], 1])  # [[0--12],[0--12]] shape:（13,13）
    # [0*13, 1*13 .... 12*13]-->  [0...0,1...1,2...2,....,12...12]
    conv_width_index = K.flatten(K.transpose(conv_width_index))  # shape(13*13，)

    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))  # (2,13*13)---transpose--->(13*13,2)
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])  # (1, 13, 13, 1, 2）
    conv_index = K.cast(conv_index, K.dtype(feats))  # 数据类型转化

    # feats：（batch,13,13,260）--->(batch,13,13,5,(47+5))
    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, clsnum + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))  # (1, 1, 1, 1, 2)

    box_xy = K.sigmoid(feats[..., :2])  # 网络输出，xy编码
    box_wh = K.exp(feats[..., 2:4])  # 网络输出，wh编码
    box_confidence = K.sigmoid(feats[..., 4:5])  # 置信度编码
    box_class_probs = K.softmax(feats[..., 5:])  # 类别预测输出

    box_xy = (box_xy + conv_index) / conv_dims  # 网络输出加上gride的位置之后相对于特征图进行归一化
    box_wh = box_wh * anchors_tensor / conv_dims
    # (batch,13,13,5,2),(batch,13,13,5,2),(batch,13,13,5,1),(batch,13,13,5,47)
    return box_xy, box_wh, box_confidence, box_class_probs


if __name__ == "__main__":
    model = tf.keras.models.load_model("./logs/new_2.h5", compile=False,
                                       custom_objects={'SpaceToDepth': SpaceToDepth(layers.Layer),
                                                       'yolo_head': yolo_head, 'yolo_choose': yolo_choose})
    model.summary()
    # x_tensor_spec = tf.TensorSpec(shape=[None, 416, 416, 3], dtype=tf.float32)
    # y_tensor_spec = tf.TensorSpec(shape=[None, None,5], dtype=tf.float32)
    # z_tensor_spec = tf.TensorSpec(shape=[None, 13, 13, 5, 1], dtype=tf.float32)
    # e_tensor_spec = tf.TensorSpec(shape=[None, 13, 13, 5, 5], dtype=tf.float32)
    full_model = tf.function(lambda *input: model(input))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./trans_model",
                      name="yad2k_new2.pb",
                      as_text=False)
