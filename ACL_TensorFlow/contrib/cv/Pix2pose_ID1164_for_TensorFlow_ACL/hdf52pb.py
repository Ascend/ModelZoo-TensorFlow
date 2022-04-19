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
# ===========================
#   Author      : Ma Shiyuan
#   Time        : 2022/4
#   Language    : Python
# ===========================
import os,sys
os.system("pip install keras==2.2.4")
print("-----------------------------------------")

from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util, graph_io
##keras库保存模型包括
#load_model：模型和参数
#load_weights：仅保存参数
#本代码实现：从A.hdf5（包括模型和参数，由load_model保存获得）中提取模型，从B.hdf5（仅包括参数，由load_weights保存获得）中提取参数，将B中的参数塞到A模型中
#必须保持A和B模型一致
#将上述hdf5文件转化为pb文件

def h5_to_pb(h5_weight_path, output_dir, out_prefix="output_", log_tensorboard=True):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    h5_model = build_model()
    h5_model.load_weights(h5_weight_path)

    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))

    model_name = os.path.splitext(os.path.split(h5_weight_path)[-1])[0] + index + '.pb'

    sess = K.get_session()
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(os.path.join(output_dir, model_name), output_dir)


def build_model():
    h5_model = load_model(inference_model_hdf5)
    return h5_model


if __name__ == '__main__':
    output_dir = os.path.join(output_path)
    h5_weight_path=os.path.join(inference_weight_hdf5)
    h5_to_pb(h5_weight_path, output_dir)
    print('finished')