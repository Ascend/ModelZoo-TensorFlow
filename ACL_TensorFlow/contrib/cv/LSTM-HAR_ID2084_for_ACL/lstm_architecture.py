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

__author__ = 'gchevalier'

import tensorflow as tf
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
import os
import sys
import argparse
from npu_bridge.estimator import npu_ops
import time

from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.core.protobuf import config_pb2


def get_config(args):
    """
    This function parses the command line arguments
    Args:
        args(str) : The command line arguments

    Returns
        args (dict): The arguments parsed into a dictionary
    """

    parser = argparse.ArgumentParser(description='Experiment parameters')
    parser.add_argument("--precision_mode", default='allow_mix_precision',
                        help="precision mode could bu \"allow_fp32_to_fp16\" or \"force_fp16\" or \"must_keep_origin_dtype\" or \"allow_mix_precision\"")
    parser.add_argument("--loss_scale_value", type=float, help="loss scale value.")
    parser.add_argument("--loss_scale_flag", type=int, default=0, help="whether to set loss scals.")
    parser.add_argument("--over_dump", default='False', help="whether to open over dump.")

    parser.add_argument("--data_dump_flag", type=int, default=0, help="whether to training data.")
    parser.add_argument("--data_dump_step", default='0|5|10', help="data dump step.")
    parser.add_argument("--profiling", default='False', help="whether to open profiling.")
    parser.add_argument("--random_remove", default='False', help="whether to remove random op in training.")
    parser.add_argument("--data_path", default='UCI HAR Dataset', help="training input data path.")
    parser.add_argument("--output_path", default='output', help="training input data path.")

    parser.add_argument("--batch_size", type=int, help="train batch size.")
    parser.add_argument("--learing_rata", type=float, help="learning rate.")
    parser.add_argument("--steps", type=int, help="training steps")
    parser.add_argument("--ckpt_count", type=int, help="save checkpoiont max counts.")
    parser.add_argument("--epochs", type=int, help="epoch number.")

    args, unknown = parser.parse_known_args(args)

    return args


args = get_config(sys.argv[1:])
print("-------")
print(args)

def npu_session_config_init(args, session_config=None):
    """
    This function config npu session
    Args:
        args: The input paras
        session_config: npu session config
    Returns
        session_config
    """
    session_config = tf.ConfigProto(log_device_placement=False)

    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["use_off_line"].b = True
    #calc mode
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

    #profiling config
    custom_op.parameter_map["profiling_mode"].b = True
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/ma-user/modelarts/outputs/train_url_0/", "training_trace":"on", "task_trace":"on", "aicpu":"on", "fp_point":"", "bp_point":""}')

    #data_dump
    if args.data_dump_flag == 1:
        data_dump_path = os.path.join(args.output_path, "data_dump")
        if (not os.path.exists(data_dump_path)):
            os.mkdir(data_dump_path)

        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(data_dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(args.data_dump_step)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

    #over dump
    if args.over_dump == 'True':
        over_dump_path = os.path.join(args.output_path, "over_dump")
        if (not os.path.exists(over_dump_path)):
            os.mkdir(over_dump_path)

        custom_op.parameter_map["enable_dump_debug"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(over_dump_path)
        custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")

    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    return session_config

def one_hot(y):
    """convert label from dense to one hot
      argument:
        label: ndarray dense label ,shape: [sample_num,1]
      return:
        one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
    """
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y = y.reshape(len(y))
    n_values = np.max(y) + 1
    return np.eye(n_values)[np.array(y, dtype=np.int32)]  # Returns FLOATS


def batch_norm(input_tensor, config, i):
    # Implementing batch normalisation: this is used out of the residual layers
    # to normalise those output neurons by mean and standard deviation.

    if config.n_layers_in_highway == 0:
        # There is no residual layers, no need for batch_norm:
        return input_tensor

    with tf.variable_scope("batch_norm") as scope:
        if i != 0:
            # Do not create extra variables for each time step
            scope.reuse_variables()

        # Mean and variance normalisation simply crunched over all axes
        axes = list(range(len(input_tensor.get_shape())))

        mean, variance = tf.nn.moments(input_tensor, axes=axes, shift=None, name=None, keep_dims=False)
        stdev = tf.sqrt(variance+0.001)

        # Rescaling
        bn = input_tensor - mean
        bn /= stdev
        # Learnable extra rescaling

        # tf.get_variable("relu_fc_weights", initializer=tf.random_normal(mean=0.0, stddev=0.0)
        bn *= tf.get_variable("a_noreg", initializer=tf.random_normal([1], mean=0.5, stddev=0.0))
        bn += tf.get_variable("b_noreg", initializer=tf.random_normal([1], mean=0.0, stddev=0.0))
        # bn *= tf.Variable(0.5, name=(scope.name + "/a_noreg"))
        # bn += tf.Variable(0.0, name=(scope.name + "/b_noreg"))

    return bn

def relu_fc(input_2D_tensor_list, features_len, new_features_len, config):
    """make a relu fully-connected layer, mainly change the shape of tensor
       both input and output is a list of tensor
        argument:
            input_2D_tensor_list: list shape is [batch_size,feature_num]
            features_len: int the initial features length of input_2D_tensor
            new_feature_len: int the final features length of output_2D_tensor
            config: Config used for weights initializers
        return:
            output_2D_tensor_list lit shape is [batch_size,new_feature_len]
    """

    W = tf.get_variable(
        "relu_fc_weights",
        initializer=tf.random_normal(
            [features_len, new_features_len],
            mean=0.0,
            stddev=float(config.weights_stddev)
        )
    )
    b = tf.get_variable(
        "relu_fc_biases_noreg",
        initializer=tf.random_normal(
            [new_features_len],
            mean=float(config.bias_mean),
            stddev=float(config.weights_stddev)
        )
    )

    # intra-timestep multiplication:
    output_2D_tensor_list = [
        tf.nn.relu(tf.matmul(input_2D_tensor, W) + b)
            for input_2D_tensor in input_2D_tensor_list
    ]

    return output_2D_tensor_list


def single_LSTM_cell(input_hidden_tensor, n_outputs):
    """ define the basic LSTM layer
        argument:
            input_hidden_tensor: list a list of tensor,
                                 shape: time_steps*[batch_size,n_inputs]
            n_outputs: int num of LSTM layer output
        return:
            outputs: list a time_steps list of tensor,
                     shape: time_steps*[batch_size,n_outputs]
    """
    with tf.variable_scope("lstm_cell"):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_outputs, state_is_tuple=True, forget_bias=0.999)
        outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, input_hidden_tensor, dtype=tf.float32)
    return outputs


def bi_LSTM_cell(input_hidden_tensor, n_inputs, n_outputs, config):
    """build bi-LSTM, concatenating the two directions in an inner manner.
        argument:
            input_hidden_tensor: list a time_steps series of tensor, shape: [sample_num, n_inputs]
            n_inputs: int units of input tensor
            n_outputs: int units of output tensor, each bi-LSTM will have half those internal units
            config: Config used for the relu_fc
        return:
            layer_hidden_outputs: list a time_steps series of tensor, shape: [sample_num, n_outputs]
    """
    n_outputs = int(n_outputs/2)

    print("bidir:")

    with tf.variable_scope('pass_forward') as scope2:
        hidden_forward = relu_fc(input_hidden_tensor, n_inputs, n_outputs, config)
        forward = single_LSTM_cell(hidden_forward, n_outputs)

    print (len(hidden_forward), str(hidden_forward[0].get_shape()))

    # Backward pass is as simple as surrounding the cell with a double inversion:
    with tf.variable_scope('pass_backward') as scope2:
        hidden_backward = relu_fc(input_hidden_tensor, n_inputs, n_outputs, config)
        backward = list(reversed(single_LSTM_cell(list(reversed(hidden_backward)), n_outputs)))

    with tf.variable_scope('bidir_concat') as scope:
        # Simply concatenating cells' outputs at each timesteps on the innermost
        # dimension, like if the two cells acted as one cell
        # with twice the n_hidden size:
        layer_hidden_outputs = [
            tf.concat(len(f.get_shape()) - 1, [f, b])
                for f, b in zip(forward, backward)]

    return layer_hidden_outputs


def residual_bidirectional_LSTM_layers(input_hidden_tensor, n_input, n_output, layer_level, config):
    """This architecture is only enabled if "config.n_layers_in_highway" has a
    value only greater than int(0). The arguments are same than for bi_LSTM_cell.
    arguments:
        input_hidden_tensor: list a time_steps series of tensor, shape: [sample_num, n_inputs]
        n_inputs: int units of input tensor
        n_outputs: int units of output tensor, each bi-LSTM will have half those internal units
        config: Config used for determining if there are residual connections and if yes, their number and with some batch_norm.
    return:
        layer_hidden_outputs: list a time_steps series of tensor, shape: [sample_num, n_outputs]
    """
    with tf.variable_scope('layer_{}'.format(layer_level)) as scope:

        if config.use_bidirectionnal_cells:
            get_lstm = lambda input_tensor: bi_LSTM_cell(input_tensor, n_input, n_output, config)
        else:
            get_lstm = lambda input_tensor: single_LSTM_cell(relu_fc(input_tensor, n_input, n_output, config), n_output)
        def add_highway_redisual(layer, residual_minilayer):
            return [a + b for a, b in zip(layer, residual_minilayer)]

        hidden_LSTM_layer = get_lstm(input_hidden_tensor)
        # Adding K new (residual bidir) connections to this first layer:
        for i in range(config.n_layers_in_highway - 1):
            with tf.variable_scope('LSTM_residual_{}'.format(i)) as scope2:
                hidden_LSTM_layer = add_highway_redisual(
                    hidden_LSTM_layer,
                    get_lstm(input_hidden_tensor)
                )

        # if config.also_add_dropout_between_stacked_cells:
        #     hidden_LSTM_layer = [npu_ops.dropout(out, keep_prob_for_dropout) for out in hidden_LSTM_layer]

        return [batch_norm(out, config, i) for i, out in enumerate(hidden_LSTM_layer)]


def LSTM_network(feature_mat, config):
    """model a LSTM Network,
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : ndarray  output shape [batch_size, n_classes]
    """

    with tf.variable_scope('LSTM_network') as scope:  # TensorFlow graph naming

        # feature_mat = npu_ops.dropout(feature_mat, keep_prob_for_dropout)

        # Exchange dim 1 and dim 0
        feature_mat = tf.transpose(feature_mat, [1, 0, 2])
        print(feature_mat.get_shape())
        # New feature_mat's shape: [time_steps, batch_size, n_inputs]

        # Temporarily crush the feature_mat's dimensions
        feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])
        print(feature_mat.get_shape())
        # New feature_mat's shape: [time_steps*batch_size, n_inputs]

        # Split the series because the rnn cell needs time_steps features, each of shape:
        hidden = tf.split(feature_mat, config.n_steps, axis=0)
        print (len(hidden), str(hidden[0].get_shape()))
        # New shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_hidden]

        # Stacking LSTM cells, at least one is stacked:
        print("\nCreating hidden #1:")
        hidden = residual_bidirectional_LSTM_layers(hidden, config.n_inputs, config.n_hidden, 1, config)
        print (len(hidden), str(hidden[0].get_shape()))

        for stacked_hidden_index in range(config.n_stacked_layers - 1):
            # If the config permits it, we stack more lstm cells:
            print("\nCreating hidden #{}:".format(stacked_hidden_index+2))
            hidden = residual_bidirectional_LSTM_layers(hidden, config.n_hidden, config.n_hidden, stacked_hidden_index+2, config)
            print (len(hidden), str(hidden[0].get_shape()))

        print("")

        # Final fully-connected activation logits
        # Get the last output tensor of the inner loop output series, of shape [batch_size, n_classes]
        # last_hidden = npu_ops.dropout(hidden[-1], keep_prob_for_dropout)
        last_logits = relu_fc(
            [hidden[-1]],
            config.n_hidden, config.n_classes, config
        )[0]
        return last_logits


def run_with_config(Config, X_train, y_train, X_test, y_test):
    e2e_start_time = time.time()

    tf.reset_default_graph()  # To enable to run multiple things in a loop

    #-----------------------------------
    # Define parameters for model
    #-----------------------------------
    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    #------------------------------------------------------
    # Let's get serious and build the neural network
    #------------------------------------------------------
    # with tf.device("npu"):  # Remove this line to use GPU. If you have a too small GPU, it crashes.
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs], name="X")
    Y = tf.placeholder(tf.float32, [None, config.n_classes], name="Y")

    # is_train for dropout control:
    is_train = tf.placeholder(tf.bool, name="is_train")
    keep_prob_for_dropout = tf.cond(is_train,
        lambda: tf.constant(
            config.keep_prob_for_dropout,
            name="keep_prob_for_dropout"
        ),
        lambda: tf.constant(
            1.0,
            name="keep_prob_for_dropout"
        )
    )

    pred_y = LSTM_network(X, config, keep_prob_for_dropout)

    # Loss, optimizer, evaluation

    # Softmax loss with L2 and L1 layer-wise regularisation
    print("Unregularised variables:")
    for unreg in [tf_var.name for tf_var in tf.trainable_variables() if ("noreg" in tf_var.name or "Bias" in tf_var.name)]:
        print(unreg)
    l2 = config.lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables()
            if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
    )
    # first_weights = [w for w in tf.all_variables() if w.name == 'LSTM_network/layer_1/pass_forward/relu_fc_weights:0'][0]
    # l1 = config.lambda_loss_amount * tf.reduce_mean(tf.abs(first_weights))
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=Y, logits=pred_y)) + l2  # + l1

    # Gradient clipping Adam optimizer with gradient noise
    optimize = tf.contrib.layers.optimize_loss(
        loss,
        global_step=tf.Variable(0),
        learning_rate=config.learning_rate,
        optimizer=tf.train.AdamOptimizer(learning_rate=config.learning_rate),
        clip_gradients=config.clip_gradients,
        gradient_noise_scale=config.gradient_noise_scale
    )

    correct_pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    #--------------------------------------------
    # Hooray, now train the neural network
    #--------------------------------------------
    # Note that log_device_placement can be turned of for less console spam.
    with tf.Session(config=npu_session_config_init(args)) as sess:
        tf.initialize_all_variables().run()

        best_accuracy = (0.0, "iter: -1")
        best_f1_score = (0.0, "iter: -1")

        # Start training for each batch and loop epochs

        worst_batches = []

        for i in range(config.training_epochs):

            # Loop batches for an epoch:
            shuffled_X, shuffled_y = shuffle(X_train, y_train, random_state=i*42)
            for start, end in zip(range(0, config.train_count, config.batch_size),
                                  range(config.batch_size, config.train_count + 1, config.batch_size)):
                start_time = time.time()
                _, train_acc, train_loss, train_pred = sess.run(
                    [optimize, accuracy, loss, pred_y],
                    feed_dict={
                        X: shuffled_X[start:end],
                        Y: shuffled_y[start:end],
                        is_train: True
                    }
                )
                cost_time = time.time() - start_time
                print("epoch : {}----step : {}----loss : {}----sec/step : {}"
                      .format(i, int(start/config.batch_size), train_loss, cost_time))

                worst_batches.append(
                    (train_loss, shuffled_X[start:end], shuffled_y[start:end])
                )
                worst_batches = list(sorted(worst_batches))[-5:]  # Keep 5 poorest

            # Train F1 score is not on boosting
            train_f1_score = metrics.f1_score(
                shuffled_y[start:end].argmax(1), train_pred.argmax(1), average="weighted"
            )

            # Retrain on top worst batches of this epoch (boosting):
            # a.k.a. "focus on the hardest exercises while training":
            for _, x_, y_ in worst_batches:

                _, train_acc, train_loss, train_pred = sess.run(
                    [optimize, accuracy, loss, pred_y],
                    feed_dict={
                        X: x_,
                        Y: y_,
                        is_train: True
                    }
                )

            # Test completely at the end of every epoch:
            # Calculate accuracy and F1 score
            pred_out, accuracy_out, loss_out = sess.run(
                [pred_y, accuracy, loss],
                feed_dict={
                    X: X_test,
                    Y: y_test,
                    is_train: False
                }
            )

            # "y_test.argmax(1)": could be optimised by being computed once...
            f1_score_out = metrics.f1_score(
                y_test.argmax(1), pred_out.argmax(1), average="weighted"
            )

            print (
                "iter: {}, ".format(i) + \
                "train loss: {}, ".format(train_loss) + \
                "train accuracy: {}, ".format(train_acc) + \
                "train F1-score: {}, ".format(train_f1_score) + \
                "test loss: {}, ".format(loss_out) + \
                "test accuracy: {}, ".format(accuracy_out) + \
                "test F1-score: {}".format(f1_score_out)
            )

            best_accuracy = max(best_accuracy, (accuracy_out, "iter: {}".format(i)))
            best_f1_score = max(best_f1_score, (f1_score_out, "iter: {}".format(i)))

        print("")
        print("final test accuracy: {}".format(accuracy_out))
        print("best epoch's test accuracy: {}".format(best_accuracy))
        print("final F1 score: {}".format(f1_score_out))
        print("best epoch's F1 score: {}".format(best_f1_score))
        print("")

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        saver.save(sess, os.path.join(args.output_path, "mode.ckpt"))
        tf.io.write_graph(sess.graph, args.output_path, 'graph.pbtxt', as_text=True)

        e2e_cost_time = time.time() - e2e_start_time

        with open(os.path.join(args.output_path, "performance_precision.txt"), "w") as file_write:
            write_str = "Final Accuracy accuracy : " + str(round(accuracy_out, 4))
            print(str(write_str))
            file_write.write(write_str)
            file_write.write('\r\n')

            write_str = "Final Performance ms/step : " + str(round(cost_time * 1000, 4))
            print(str(write_str))
            file_write.write(write_str)
            file_write.write('\r\n')

            write_str = "Final Training Duration sec : " + str(round(e2e_cost_time, 4))
            print(str(write_str))
            file_write.write(write_str)
            file_write.write('\r\n')

    # returning both final and bests accuracies and f1 scores.
    return accuracy_out, best_accuracy, f1_score_out, best_f1_score