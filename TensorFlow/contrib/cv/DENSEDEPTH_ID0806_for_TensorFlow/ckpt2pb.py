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
import argparse
import tensorflow as tf
from npu_bridge.npu_init import *
from model import create_model
import os

sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["graph_run_mode"].i = 0

sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

npu_keras_sess = set_keras_session_npu_config(config=sess_config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--ckptdir', type=str, default='./ckpt_npu', help='ckpt files input path .')
parser.add_argument('--pbdir', type=str, default='./result/pb', help='pb files output path.')
parser.add_argument('--output_node_names', type=str, default='conv3/BiasAdd', help='output node names.')
parser.add_argument('--pb_name', type=str, default='test', help='pb file name.')
args = parser.parse_args()


def keras_model_to_frozen_graph():
    """ convert ckpt model file to frozen graph(.pb file)
    """
    import tensorflow as tf
    from tensorflow.python.framework import graph_io

    def freeze_graph(graph, session, output_node_names, model_name, output_path):
        with graph.as_default():
            graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
            # for node in graphdef_inf.node:
            #     print('node:', node.name)
            graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output_node_names)
            graph_io.write_graph(graphdef_frozen, output_path, os.path.basename(model_name) + ".pb", as_text=False)
            print('*'*20)
            print("pb file in:{}".format(output_path))
            print('*' * 20)

            # output_graph = os.path.basename(model_name) + ".pb"
            # with tf.gfile.GFile(output_graph, "wb") as f:
            #     f.write(graphdef_frozen.SerializeToString())

    tf.keras.backend.set_learning_phase(0)  # this line most important
    model = create_model()
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(
        args.ckptdir)).assert_existing_objects_matched().expect_partial().run_restore_ops()
    session = tf.keras.backend.get_session()
    # freeze_graph(session.graph, session, ["conv3/bias"], model_name)
    # print("out.op.name:", [out.op.name for out in model.outputs])
    freeze_graph(session.graph, session, [out.op.name for out in model.outputs], args.pb_name, args.pbdir)


keras_model_to_frozen_graph()

close_session(npu_keras_sess)
