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
import sys
import os
from npu_bridge.npu_init import *
import argparse

from lstm_architecture import one_hot, run_with_config


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

#--------------------------------------------
# Neural net's config.
#--------------------------------------------

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, X_train, X_test):
        # Data shaping
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series
        self.n_classes = 6  # Final output classes

        # Training
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.005
        self.training_epochs = 250  # 250
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
        self.n_inputs = len(X_train[0][0])
        self.n_hidden = 28  # nb of neurons inside the neural network
        # Use bidir in every LSTM cell, or not:
        self.use_bidirectionnal_cells = False

        # High-level deep architecture
        self.also_add_dropout_between_stacked_cells = False  # True
        # NOTE: values of exactly 1 (int) for those 2 high-level parameters below totally disables them and result in only 1 starting LSTM.
        # self.n_layers_in_highway = 1  # Number of residual connections to the LSTMs (highway-style), this is did for each stacked block (inside them).
        # self.n_stacked_layers = 1  # Stack multiple blocks of residual
        # layers.


#--------------------------------------------
# Dataset-specific constants and functions + loading
#--------------------------------------------

# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    """
    Given attribute (train or test) of feature, read all 9 features into an
    np ndarray of shape [sample_sequence_idx, time_step, feature_num]
        argument:   X_signals_paths str attribute of feature: 'train' or 'test'
        return:     np ndarray, tensor of features
    """
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train_signals_paths = [
        args.data_path + "train/Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
        args.data_path + "test/Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]
X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    """
    Read Y file of values to be predicted
        argument: y_path str attibute of Y: 'train' or 'test'
        return: Y ndarray / tensor of the 6 one_hot labels of each sample
    """
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return one_hot(y_ - 1)

y_train_path = args.data_path + "train/" + "y_train.txt"
y_test_path = args.data_path + "test/" + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)


#--------------------------------------------
# Training (maybe multiple) experiment(s)
#--------------------------------------------

n_layers_in_highway = 0
n_stacked_layers = 3
trial_name = "{}x{}".format(n_layers_in_highway, n_stacked_layers)

for learning_rate in [0.001]:  # [0.01, 0.007, 0.001, 0.0007, 0.0001]:
    for lambda_loss_amount in [0.005]:
        for clip_gradients in [15.0]:
            print("learning_rate: {}".format(learning_rate))
            print("lambda_loss_amount: {}".format(lambda_loss_amount))
            print("")

            class EditedConfig(Config):
                def __init__(self, X, Y):
                    super(EditedConfig, self).__init__(X, Y)

                    if args.batch_size is not None and args.batch_size > 0:
                        self.batch_size = args.batch_size

                    if args.epochs is not None and args.epochs > 0:
                        self.training_epochs = args.epochs

                    # Edit only some parameters:
                    self.learning_rate = learning_rate
                    self.lambda_loss_amount = lambda_loss_amount
                    self.clip_gradients = clip_gradients
                    # Architecture params:
                    self.n_layers_in_highway = n_layers_in_highway
                    self.n_stacked_layers = n_stacked_layers

            # # Useful catch upon looping (e.g.: not enough memory)
            # try:
            #     accuracy_out, best_accuracy = run_with_config(EditedConfig)
            # except:
            #     accuracy_out, best_accuracy = -1, -1
            accuracy_out, best_accuracy, f1_score_out, best_f1_score = (
                run_with_config(EditedConfig, X_train, y_train, X_test, y_test)
            )
            print (accuracy_out, best_accuracy, f1_score_out, best_f1_score)


            with open('{}_result_HAR_6.txt'.format(trial_name), 'a') as f:
                f.write(str(learning_rate) + ' \t' + str(lambda_loss_amount) + ' \t' + str(clip_gradients) + ' \t' + str(
                    accuracy_out) + ' \t' + str(best_accuracy) + ' \t' + str(f1_score_out) + ' \t' + str(best_f1_score) + '\n\n')

            print("________________________________________________________")
        print("")
print("Done.")
