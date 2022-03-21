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
import time
import argparse
import numpy as np
import tensorflow as tf

import preprocess.mnist as preprocess
import utils
from model import model_utils
from model import imm


print("==> parsing input arguments")
flags = tf.app.flags

## Data input settings
flags.DEFINE_boolean("mean_imm", True, "include Mean-IMM")
flags.DEFINE_boolean("mode_imm", True, "include Mode-IMM")

## Model Hyperparameter 
flags.DEFINE_float("regularizer", -1, "L2 Regularization parameter")
flags.DEFINE_float("alpha", -1, "alpha(K) of Mean & Mode IMM (cf. equation (3)~(8) in the article)")

## Training Hyperparameter
flags.DEFINE_float("epoch", -1, "the number of training epoch")
flags.DEFINE_string("optimizer", 'SGD', "the method name of optimization. (SGD|Adam|Momentum)")
flags.DEFINE_float("learning_rate", -1, "learning rate of optimizer")
flags.DEFINE_integer("batch_size", 50, "mini batch size")

FLAGS = flags.FLAGS
utils.SetDefaultAsNatural(FLAGS)


mean_imm = FLAGS.mean_imm
mode_imm = FLAGS.mode_imm
lmbda = FLAGS.regularizer
alpha = FLAGS.alpha
optimizer = FLAGS.optimizer
learning_rate = FLAGS.learning_rate
epoch = int(FLAGS.epoch)
batch_size = FLAGS.batch_size

no_of_task = 3
no_of_node = [784,800,800,10]
keep_prob_info = [0.8, 0.5, 0.5]


# data preprocessing
x, y, x_, y_, xyc_info = preprocess.XycPackage()

start = time.time()

with tf.Session(config=npu_config_proto(config_proto=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))) as sess:

    mlp = imm.TransferNN(no_of_node, (optimizer, learning_rate), keep_prob_info=keep_prob_info)
    mlp.RegPatch(lmbda)

    sess.run(tf.global_variables_initializer())
    L_copy = []
    FM = []
    for i in range(no_of_task):
        print("")
        print("================= Train task #%d (%s) ================" % (i+1, optimizer))

        if i > 0: 
            model_utils.CopyLayers(sess, mlp.Layers, mlp.Layers_reg)    # Regularization from weight of pre-task

        mlp.Train(sess, x[i], y[i], np.concatenate(x_), np.concatenate(y_), epoch, mb=batch_size)
        mlp.Test(sess, [[x[i],y[i]," train"], [x_[i],y_[i]," test"]])

        if mean_imm or mode_imm:
            L_copy.append(model_utils.CopyLayerValues(sess, mlp.Layers))
        if mode_imm:
            FM.append(mlp.CalculateFisherMatrix(sess, x[i], y[i]))

    mlp.TestAllTasks(sess, x_, y_)

    alpha_list = [(1-alpha)/(no_of_task-1)] * (no_of_task-1)
    alpha_list.append(alpha)
    print(alpha_list)
    ######################### Mean-IMM ##########################
    if mean_imm:
        print("")
        print("Main experiment on L2-transfer + Mean-IMM, shuffled MNIST")
        print("============== Train task #%d (Mean-IMM) ==============" % no_of_task)

        LW = model_utils.UpdateMultiTaskLwWithAlphas(L_copy[0], alpha_list, no_of_task)
        model_utils.AddMultiTaskLayers(sess, L_copy, mlp.Layers, LW, no_of_task)
        ret = mlp.TestTasks(sess, x, y, x_, y_, debug=False)
        utils.PrintResults(alpha, ret)

        mlp.TestAllTasks(sess, x_, y_)

    ######################### Mode-IMM ##########################
    if mode_imm:
        print("")
        print("Main experiment on L2-transfer + Mode-IMM, shuffled MNIST")
        print("============== Train task #%d (Mode-IMM) ==============" % no_of_task)

        LW = model_utils.UpdateMultiTaskWeightWithAlphas(FM, alpha_list, no_of_task)
        model_utils.AddMultiTaskLayers(sess, L_copy, mlp.Layers, LW, no_of_task)
        ret = mlp.TestTasks(sess, x, y, x_, y_, debug=False)
        utils.PrintResults(alpha, ret)

        mlp.TestAllTasks(sess, x_, y_)

    print("")
    print("Time: %.4f s" % (time.time()-start))

