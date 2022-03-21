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

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import os, sys
import argparse

base_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_path + "/../")

from lstm_architecture import LSTM_network, residual_bidirectional_LSTM_layers, relu_fc

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):
        # Data shaping
        self.train_count = 7352  # 7352 training series
        self.test_data_count = 2947  # 2947 testing series
        self.n_steps = 128  # 128 time_steps per series
        self.n_classes = 6  # Final output classes

        # Training
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.005
        self.training_epochs = 20
        self.batch_size = 100
        self.clip_gradients = 15.0
        self.gradient_noise_scale = None
        # Dropout is added on inputs and after each stacked layers (but not
        # between residual layers).
        self.keep_prob_for_dropout = 0.85  # **(1/3.0)

        # Linear+relu structure
        self.bias_mean = 0.3
        # I would recommend between 0.1 and 1.0 or to change and use a xavier
        # initializer
        self.weights_stddev = 0.2

        ########
        # NOTE: I think that if any of the below parameters are changed,
        # the best is to readjust every parameters in the "Training" section
        # above to properly compare the architectures only once optimised.
        ########

        # LSTM structure
        # Features count is of 9: three 3D sensors features over time
        self.n_inputs = 9
        self.n_hidden = 28  # nb of neurons inside the neural network
        # Use bidir in every LSTM cell, or not:
        self.use_bidirectionnal_cells = False

        # High-level deep architecture
        self.also_add_dropout_between_stacked_cells = False  # True
        # NOTE: values of exactly 1 (int) for those 2 high-level parameters below totally disables them and result in only 1 starting LSTM.
        self.n_layers_in_highway = 0  # Number of residual connections to the LSTMs (highway-style), this is did for each stacked block (inside them).
        self.n_stacked_layers = 3  # Stack multiple blocks of residual
        # layers.


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path', default=1,
                        help="""set checkpoint path""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def main():
    args = parse_args()

    tf.reset_default_graph()

    config = Config()
    # set inputs node
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs], name="X")

    # is_train = tf.constant(False, dtype=tf.bool)
    # keep_prob_for_dropout = tf.cond(is_train,
    #     lambda: tf.constant(
    #         config.keep_prob_for_dropout,
    #         name="keep_prob_for_dropout"
    #     ),
    #     lambda: tf.constant(
    #         1.0,
    #         name="keep_prob_for_dropout"
    #     )
    # )

    pred_y = LSTM_network(X, config)
    # tf.identity(pred_y, name='output')
    # prediction = tf.argmax(pred_y, 1, name="output")
    prediction = tf.argmax(input=pred_y, axis=-1, output_type=tf.dtypes.int32, name="output")
    print(prediction.shape)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_graph="lstm_har.pb"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        saver.restore(sess, args.ckpt_path)

        output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=["output"])

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    print("done")

if __name__ == '__main__':
    main()