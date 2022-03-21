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
"""
Generate *.pb file
"""


import moxing as mox
import argparse
import tensorflow as tf
from tcn import TCN


def parse_args():
    """Add some parameters"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--h5_obs_path', default='obs://kyq/mytestnew/mytestnew1h5/tcn.h5', type=str,
                        help='h5 obs path')
    parser.add_argument('--h5_path', default='/cache/h5/tcn.h5', type=str, help='h5 path')
    parser.add_argument('--pb_obs_path', default='obs://kyq/my/tcn.pb', type=str, help='pb obs path')
    args, unknown_args = parser.parse_known_args()
    return args


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freeze variables and convert a generator networks to a GraphDef files.
    This makes file size smaller and can be used for inference in production.
    """
    # from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                                    output_names, freeze_var_names)
        return frozen_graph


args = parse_args()

# configure paths
mox.file.copy(args.h5_obs_path, args.h5_path)
h5_model_path = args.h5_path  
output_path = '/cache/pb'
pb_model_name = 'tcn.pb'  

# import keras model
tf.keras.backend.set_learning_phase(0)
net_model = tf.keras.models.load_model(h5_model_path, custom_objects={'TCN': TCN})

# save model as .pb
sess = tf.keras.backend.get_session()
frozen_graph = freeze_session(sess, output_names=[net_model.output.op.name])
tf.io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)

mox.file.copy('/cache/pb/tcn.pb', args.pb_obs_path)
