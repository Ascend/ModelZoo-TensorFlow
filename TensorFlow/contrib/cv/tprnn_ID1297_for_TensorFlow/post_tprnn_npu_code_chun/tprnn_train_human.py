"""
Simple code for training an RNN for motion prediction on Human 3.6M Dataset.
Mainly adopted https://github.com/una-dinosauria/human-motion-prediction
"""
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import math
import os
import random
import sys
import time
import h5py

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import tprnn_basic_model
import tprnn_generic_model

import datetime
import argparse
#import moxing as mox  #del

import precision_tool.tf_config as npu_tf_config
import precision_tool.config as CONFIG

# print('===>>>{}'.format(os.getcwd()))
# print(os.system('env'))

#
# if 'session' in locals() and session is not None:
#     print('Close interactive session')
#     session.close()

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=npu_config_proto(config_proto=tf.ConfigProto(gpu_options=gpu_options)))

# 解析输入参数data_url
parser = argparse.ArgumentParser()
# Learning
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.005,
    help="Initial learning rate [default: 0.005]",
)
parser.add_argument(
    "--learning_rate_decay_factor",
    type=float,
    default=0.95,
    help="Learning rate is multiplied by this much. 1 means no decay.",
)
parser.add_argument(
    "--learning_rate_step",
    type=int,
    default=10000,
    help="Every this many steps, do decay.",
)
parser.add_argument(
    "--max_gradient_norm", type=float, default=5, help="Clip gradients to this norm."
)
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size to use during training."
)
parser.add_argument(
    "--iterations", type=int, default=int(1e5), help="Iterations to train for."
)
# Architecture
parser.add_argument(
    "--rnn_size", type=int, default=1024, help="Size of each model layer."
)
parser.add_argument(
    "--seq_length_in",
    type=int,
    default=50,
    help="Number of frames to feed into the encoder. 25 fps",
)
parser.add_argument(
    "--seq_length_out",
    type=int,
    default=10,
    help="Number of frames that the decoder has to predict. 25fps",
)
parser.add_argument(
    "--omit_one_hot",
    type=bool,
    default=True,
    help="Whether to remove one-hot encoding from the data",
)
# Directories
#tf.app.flags.DEFINE_string("data_dir", os.path.normpath("./data/h3.6m/dataset"), "Data directory")
#tf.app.flags.DEFINE_string("train_dir", os.path.normpath("./experiments/"), "Training directory.")
parser.add_argument("--action", default="all")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--save_every", type=int, default=1000)
parser.add_argument("--sample", type=bool, default=False)
parser.add_argument("--use_cpu", type=bool, default=False)
parser.add_argument("--load", type=int, default=0)

# New model paramteres
parser.add_argument("--dataset", default="human")
parser.add_argument("--tprnn_scale", type=int, default=2)
parser.add_argument("--tprnn_layers", type=int, default=2)
parser.add_argument("--more", type=int, default=0)
parser.add_argument("--dropout_keep", type=float, default=1.0)
parser.add_argument("--model", default="basic")


# parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument(
    "--train_url", type=str, default="./output"
)  # PyCharm插件传入的 OBS Path路径
parser.add_argument(
    "--data_url", type=str, default="./dataset"
)  # PyCharm插件传入的 Data Path in OBS路径

#add
parser.add_argument("--locals_dir", default="/data2/NRE_Check/wx1056345/ID1297_tprnn/data/h3.6m/dataset")
parser.add_argument("--model_dir", default="/data2/NRE_Check/wx1056345/ID1297_tprnn/result")

FLAGS = parser.parse_args()
# 在ModelArts容器创建数据存放目录
#locals_dir="/data2/NRE_Check/wx1056345/ID1297_tprnn/data/h3.6m/dataset"  #-------------------del
#locals_dir = "/cache/dataset"                                   #del
#start = datetime.datetime.now()                                 #del
#os.makedirs(locals_dir)                                         #del
# OBS数据拷贝到ModelArts容器内
#mox.file.copy_parallel(FLAGS.data_url, locals_dir)              #del
#end = datetime.datetime.now()                                   #del
#print("copy over, time:{}(s)".format((end - start).seconds))    #del

# 在ModelArts容器创建训练输出目录
#del
#model_dir = "/cache/result"
#os.makedirs(model_dir)
#model_dir="/data2/NRE_Check/wx1056345/ID1297_tprnn/result"    #----------------del

#del
#profiling_dir = "/cache/profiling"
#os.makedirs(profiling_dir)

# dump_data_dir = "/cache/dump_data"
# os.makedirs(dump_data_dir)
#
# dump_graph_dir = "/cache/dump_graph"
# os.makedirs(dump_graph_dir)  # 创建dump图生成路径

# train_dir = os.path.normpath(os.path.join(
#   config.train_url,
#   FLAGS.dataset,
#   FLAGS.action,
#   'out_{0}'.format(FLAGS.seq_length_out),
#   'iterations_{0}'.format(FLAGS.iterations),
#   'rnn_size_{0}'.format(FLAGS.rnn_size),
#   'lr_{0}'.format(FLAGS.learning_rate)))

train_dir = FLAGS.model_dir  #/data2/NRE_Check/wx1056345/ID1297_tprnn/result #-----------------add
if FLAGS.tprnn_scale > 0:
    train_dir = os.path.normpath(
        os.path.join(train_dir, "tprnn_scale_{0}".format(FLAGS.tprnn_scale))
    )
    print("train_dir:", train_dir)
if FLAGS.tprnn_layers > 0:
    train_dir = os.path.normpath(
        os.path.join(train_dir, "tprnn_layers_{0}".format(FLAGS.tprnn_layers))
    )
train_dir = os.path.normpath(
    os.path.join(train_dir, "dropout_keep_{0}".format(FLAGS.dropout_keep))
)
if FLAGS.model == "basic":
    train_dir = os.path.normpath(os.path.join(train_dir, "basic"))
else:
    train_dir = os.path.normpath(os.path.join(train_dir, "generic"))

summaries_dir = os.path.normpath(
    os.path.join(train_dir, "log")
)  # Directory for TB summaries


def create_model(session, sampling=False):
    """Create translation model and initialize or load parameters in session."""

    from keras import backend as K

    K.set_session(session)

    if FLAGS.model == "basic":
        print("------basic-------")
        model = tprnn_basic_model.TPRNNBasicModel(
            FLAGS.dataset,
            FLAGS.seq_length_in if not sampling else 50,
            FLAGS.seq_length_out,  # if not sampling else 100,
            FLAGS.rnn_size,  # hidden layer size
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.learning_rate_decay_factor,
            summaries_dir,
            tf.float32,
            FLAGS.tprnn_scale,
            FLAGS.tprnn_layers,
            FLAGS.dropout_keep,
        )
    else:
        model = tprnn_generic_model.TPRNNGenericModel(
            FLAGS.dataset,
            FLAGS.seq_length_in if not sampling else 50,
            FLAGS.seq_length_out,  # if not sampling else 100,
            FLAGS.rnn_size,  # hidden layer size
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.learning_rate_decay_factor,
            summaries_dir,
            tf.float32,
            FLAGS.tprnn_scale,
            FLAGS.tprnn_layers,
            FLAGS.dropout_keep,
        )

    if FLAGS.load <= 0:
        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        return model

    load_dir = train_dir    #/data2/NRE_Check/wx1056345/ID1297_tprnn/result/tprnn_scale_2/tprnn_layers_2/dropout_keep_1.0/basic/
    ckpt = tf.train.get_checkpoint_state(load_dir, latest_filename="checkpoint")
    print("-----------------------------------load_dir", load_dir)

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        if FLAGS.load > 0:
            if os.path.isfile(
                os.path.join(load_dir, "checkpoint-{0}.index".format(FLAGS.load))
            ):
                ckpt_name = os.path.normpath(
                    os.path.join(
                        os.path.join(load_dir, "checkpoint-{0}".format(FLAGS.load))
                    )
                )
            else:
                raise ValueError(
                    "Asked to load checkpoint {0}, but it does not seem to exist".format(
                        FLAGS.load
                    )
                )
        else:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        print("Loading model {0}".format(ckpt_name))
        # model.saver.restore( session, ckpt.model_checkpoint_path )
        model.saver.restore(session, ckpt_name)
        return model
    else:
        print("Could not find checkpoint. Aborting.")
        raise (
            ValueError,
            "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path),
        )

    return model


def train():
    print("-----------train-----------------")
    """Train a seq2seq model on human motion"""

    actions = define_actions(FLAGS.action)

    number_of_actions = len(actions)

    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
        actions,
        FLAGS.seq_length_in,
        FLAGS.seq_length_out,
        FLAGS.locals_dir,  #add
        not FLAGS.omit_one_hot,
    )
    print("====data over======")

    # # Limit TF to take a fraction of the GPU memory
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    # device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
    config_proto = tf.ConfigProto()
    custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    #add
    custom_op.parameter_map['dynamic_input'].b = True
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    ##混合精度
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

    ## dump
    # custom_op.parameter_map["enable_dump"].b = True
    # custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/HwHiAiUser/output")
    # custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
    # custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
    # custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(dump_data_dir)

    # ## profiling
    # custom_op.parameter_map["profiling_mode"].b = True
    # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
    #     '{"output":"/cache/profiling","training_trace":"on","task_trace":"on","aicpu":"on",'
    #     '"fp_point":"mul",'
    #     '"bp_point":"gradients/AddN_225",'
    #     '"aic_metrics":"PipeUtilization"}')

    config = npu_config_proto(config_proto=config_proto)
    # config = npu_tf_config.session_dump_config(config, action='dump')
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  #--------------------------add

    with tf.Session(config=config) as sess:
        # with tf.Session(config=npu_config_proto(config_proto=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count ))) as sess:

        # === Create the model ===
        model = create_model(sess)
        model.train_writer.add_graph(sess.graph)
        print("=======Model created over==========")
        # tf.io.write_graph(sess.graph, FLAGS.train_url, 'graph.pbtxt')
        # print('ok')

        # === Read and denormalize the gt with srnn's seeds, as we'll need them
        # many times for evaluation in Euler Angles ===
        srnn_gts_euler = get_srnn_gts(
            actions,
            model,
            test_set,
            data_mean,
            data_std,
            dim_to_ignore,
            not FLAGS.omit_one_hot,
        )

        # === This is the training loop ===
        perf_list = []
        fps_list = []
        step_time, loss, val_loss = 0.0, 0.0, 0.0
        current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
        previous_losses = []

        step_time, loss = 0, 0

        # Choose the best model during training using avg_mean_errors_sum
        best_avg_mean_errors_sum = 1000000
        best_avg_mean_errors = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000]
        best_current_step = 1

        iterations_to_train = FLAGS.iterations if FLAGS.more == 0 else FLAGS.more
        print("-----------iterations_to_train:",iterations_to_train) #100000
        for _ in xrange(iterations_to_train):

            start_time = time.time()
            # print('----------:', _)
            # mox.file.copy_parallel(CONFIG.ROOT_DIR, FLAGS.train_url)

            # === Training step ===
            encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(
                train_set, not FLAGS.omit_one_hot, actions
            )
            _, step_loss, loss_summary, lr_summary = model.step(
                sess, encoder_inputs, decoder_inputs, decoder_outputs, False
            )
            model.train_writer.add_summary(loss_summary, current_step)
            model.train_writer.add_summary(lr_summary, current_step)

            # mox.file.copy_parallel(dump_data_dir, FLAGS.train_url)  # 将算子的dump数据传输至OBS
            # print('dump  data ok')
            # mox.file.copy_parallel(dump_graph_dir, FLAGS.train_url)  # 将dump图文件传输至OBS
            # print('dump graph ok')
            
            if current_step % 100 == 0:    #1000/100=10轮
                if current_step > 1: #去掉第一次的数据
                    perf = (time.time() - start_time) / 100
                    perf_list.append(perf)
                    perf_ = np.mean(perf_list)
                    print(perf_)
                    fps = FLAGS.batch_size / perf
                    fps_list.append(fps)
                    fps_ = np.mean(fps_list)
                    print(fps)
                    print("step %6i  step_loss: %6.3f  time: %6.6f fps: %6.6f" %(current_step, step_loss, perf_, fps_))
                    # mox.file.copy_parallel(profiling_dir, 's3://jida-2021/post_tprnn_npu/out-profiling/')

            step_time += (time.time() - start_time) / FLAGS.test_every        #1000 #add-100
            loss += step_loss / FLAGS.test_every
            current_step += 1

            # === step decay ===
            if current_step % FLAGS.learning_rate_step == 0:                 #10000 #add
                sess.run(model.learning_rate_decay_op)

            # Once in a while, we save checkpoint, print statistics, and run evals.
            
            #---------------------------------------------------------------- 屏蔽报错可暂时去掉验证部分 ，步数控制即可
            if current_step % FLAGS.test_every == 0:                          #1000    

                print("=== Validation with randomly chosen seeds ===")
                # === Validation with randomly chosen seeds ===
                forward_only = True

                encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(
                    test_set, not FLAGS.omit_one_hot, actions
                )
                step_loss, predicted_output, loss_summary = model.step(
                    sess, encoder_inputs, decoder_inputs, decoder_outputs, forward_only
                )
                val_loss = step_loss  # Loss book-keeping

                model.test_writer.add_summary(loss_summary, current_step)

                print("{0: <16} |".format("milliseconds"), end="")
                for ms in [80, 160, 320, 400, 560, 1000]:
                    print(" {0:5d} |".format(ms), end="")
                print("=========validation========")
                # mox.file.copy_parallel('/var/log/npu', FLAGS.train_url)

                avg_mean_errors = np.zeros([model.target_seq_len])
                print("avg_mean_errors:", avg_mean_errors)
                # === Validation with srnn's seeds ===
                for action in actions:
                    print("==============action==========")
                    # mox.file.copy_parallel('/var/log/npu', FLAGS.train_url)
                    # Evaluate the model on the test batches
                    (
                        encoder_inputs,
                        decoder_inputs,
                        decoder_outputs,
                    ) = model.get_batch_srnn(test_set, action, actions)
                    srnn_loss, srnn_poses, _, = model.step(
                        sess,
                        encoder_inputs,
                        decoder_inputs,
                        decoder_outputs,
                        True,
                        True,
                    )

                    # Denormalize the output
                    srnn_pred_expmap = data_utils.revert_output_format(
                        srnn_poses,
                        data_mean,
                        data_std,
                        dim_to_ignore,
                        actions,
                        not FLAGS.omit_one_hot,
                    )

                    # Save the errors here
                    mean_errors = np.zeros(
                        (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0])
                    )

                    # Training is done in exponential map, but the error is reported in
                    # Euler angles, as in previous work.
                    # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-247769197
                    N_SEQUENCE_TEST = 8
                    for i in np.arange(N_SEQUENCE_TEST):
                        eulerchannels_pred = srnn_pred_expmap[i]

                        # Convert from exponential map to Euler angles
                        for j in np.arange(eulerchannels_pred.shape[0]):
                            for k in np.arange(3, 97, 3):
                                eulerchannels_pred[
                                    j, k: k + 3
                                ] = data_utils.rotmat2euler(
                                    data_utils.expmap2rotmat(
                                        eulerchannels_pred[j, k: k + 3]
                                    )
                                )

                        # The global translation (first 3 entries) and global rotation
                        # (next 3 entries) are also not considered in the error, so the_key
                        # are set to zero.
                        # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
                        gt_i = np.copy(srnn_gts_euler[action][i])
                        gt_i[:, 0:6] = 0

                        # Now compute the l2 error. The following is numpy port of the error
                        # function provided by Ashesh Jain (in matlab), available at
                        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
                        idx_to_use = np.where(np.std(gt_i, 0) > 1e-4)[0]

                        euc_error = np.power(
                            gt_i[:, idx_to_use] - eulerchannels_pred[:, idx_to_use], 2
                        )
                        euc_error = np.sum(euc_error, 1)
                        euc_error = np.sqrt(euc_error)
                        mean_errors[i, :] = euc_error

                    # This is simply the mean error over the N_SEQUENCE_TEST examples
                    mean_mean_errors = np.mean(mean_errors, 0)
                    avg_mean_errors += mean_mean_errors

                    # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
                    print("{0: <16} |".format(action), end="")
                    for ms in [1, 3, 7, 9, 13, 24]:
                        if FLAGS.seq_length_out >= ms + 1:
                            print(" {0:.3f} |".format(mean_mean_errors[ms]), end="")
                        else:
                            print("   n/a |", end="")
                    print()

                    # Ugly massive if-then to log the error to tensorboard:shrug:
                    if action == "walking":
                        summaries = sess.run(
                            [
                                model.walking_err80_summary,
                                model.walking_err160_summary,
                                model.walking_err320_summary,
                                model.walking_err400_summary,
                                model.walking_err560_summary,
                                model.walking_err1000_summary,
                            ],
                            {
                                model.walking_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.walking_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.walking_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.walking_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.walking_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.walking_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "eating":
                        summaries = sess.run(
                            [
                                model.eating_err80_summary,
                                model.eating_err160_summary,
                                model.eating_err320_summary,
                                model.eating_err400_summary,
                                model.eating_err560_summary,
                                model.eating_err1000_summary,
                            ],
                            {
                                model.eating_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.eating_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.eating_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.eating_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.eating_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.eating_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "smoking":
                        summaries = sess.run(
                            [
                                model.smoking_err80_summary,
                                model.smoking_err160_summary,
                                model.smoking_err320_summary,
                                model.smoking_err400_summary,
                                model.smoking_err560_summary,
                                model.smoking_err1000_summary,
                            ],
                            {
                                model.smoking_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.smoking_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.smoking_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.smoking_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.smoking_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.smoking_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "discussion":
                        summaries = sess.run(
                            [
                                model.discussion_err80_summary,
                                model.discussion_err160_summary,
                                model.discussion_err320_summary,
                                model.discussion_err400_summary,
                                model.discussion_err560_summary,
                                model.discussion_err1000_summary,
                            ],
                            {
                                model.discussion_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.discussion_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.discussion_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.discussion_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.discussion_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.discussion_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "directions":
                        summaries = sess.run(
                            [
                                model.directions_err80_summary,
                                model.directions_err160_summary,
                                model.directions_err320_summary,
                                model.directions_err400_summary,
                                model.directions_err560_summary,
                                model.directions_err1000_summary,
                            ],
                            {
                                model.directions_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.directions_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.directions_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.directions_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.directions_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.directions_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "greeting":
                        summaries = sess.run(
                            [
                                model.greeting_err80_summary,
                                model.greeting_err160_summary,
                                model.greeting_err320_summary,
                                model.greeting_err400_summary,
                                model.greeting_err560_summary,
                                model.greeting_err1000_summary,
                            ],
                            {
                                model.greeting_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.greeting_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.greeting_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.greeting_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.greeting_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.greeting_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "phoning":
                        summaries = sess.run(
                            [
                                model.phoning_err80_summary,
                                model.phoning_err160_summary,
                                model.phoning_err320_summary,
                                model.phoning_err400_summary,
                                model.phoning_err560_summary,
                                model.phoning_err1000_summary,
                            ],
                            {
                                model.phoning_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.phoning_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.phoning_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.phoning_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.phoning_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.phoning_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "posing":
                        summaries = sess.run(
                            [
                                model.posing_err80_summary,
                                model.posing_err160_summary,
                                model.posing_err320_summary,
                                model.posing_err400_summary,
                                model.posing_err560_summary,
                                model.posing_err1000_summary,
                            ],
                            {
                                model.posing_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.posing_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.posing_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.posing_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.posing_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.posing_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "purchases":
                        summaries = sess.run(
                            [
                                model.purchases_err80_summary,
                                model.purchases_err160_summary,
                                model.purchases_err320_summary,
                                model.purchases_err400_summary,
                                model.purchases_err560_summary,
                                model.purchases_err1000_summary,
                            ],
                            {
                                model.purchases_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.purchases_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.purchases_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.purchases_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.purchases_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.purchases_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "sitting":
                        summaries = sess.run(
                            [
                                model.sitting_err80_summary,
                                model.sitting_err160_summary,
                                model.sitting_err320_summary,
                                model.sitting_err400_summary,
                                model.sitting_err560_summary,
                                model.sitting_err1000_summary,
                            ],
                            {
                                model.sitting_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.sitting_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.sitting_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.sitting_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.sitting_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.sitting_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "sittingdown":
                        summaries = sess.run(
                            [
                                model.sittingdown_err80_summary,
                                model.sittingdown_err160_summary,
                                model.sittingdown_err320_summary,
                                model.sittingdown_err400_summary,
                                model.sittingdown_err560_summary,
                                model.sittingdown_err1000_summary,
                            ],
                            {
                                model.sittingdown_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.sittingdown_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.sittingdown_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.sittingdown_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.sittingdown_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.sittingdown_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "takingphoto":
                        summaries = sess.run(
                            [
                                model.takingphoto_err80_summary,
                                model.takingphoto_err160_summary,
                                model.takingphoto_err320_summary,
                                model.takingphoto_err400_summary,
                                model.takingphoto_err560_summary,
                                model.takingphoto_err1000_summary,
                            ],
                            {
                                model.takingphoto_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.takingphoto_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.takingphoto_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.takingphoto_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.takingphoto_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.takingphoto_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "waiting":
                        summaries = sess.run(
                            [
                                model.waiting_err80_summary,
                                model.waiting_err160_summary,
                                model.waiting_err320_summary,
                                model.waiting_err400_summary,
                                model.waiting_err560_summary,
                                model.waiting_err1000_summary,
                            ],
                            {
                                model.waiting_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.waiting_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.waiting_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.waiting_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.waiting_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.waiting_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "walkingdog":
                        summaries = sess.run(
                            [
                                model.walkingdog_err80_summary,
                                model.walkingdog_err160_summary,
                                model.walkingdog_err320_summary,
                                model.walkingdog_err400_summary,
                                model.walkingdog_err560_summary,
                                model.walkingdog_err1000_summary,
                            ],
                            {
                                model.walkingdog_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.walkingdog_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.walkingdog_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.walkingdog_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.walkingdog_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.walkingdog_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )
                    elif action == "walkingtogether":
                        summaries = sess.run(
                            [
                                model.walkingtogether_err80_summary,
                                model.walkingtogether_err160_summary,
                                model.walkingtogether_err320_summary,
                                model.walkingtogether_err400_summary,
                                model.walkingtogether_err560_summary,
                                model.walkingtogether_err1000_summary,
                            ],
                            {
                                model.walkingtogether_err80: mean_mean_errors[1]
                                if FLAGS.seq_length_out >= 2
                                else None,
                                model.walkingtogether_err160: mean_mean_errors[3]
                                if FLAGS.seq_length_out >= 4
                                else None,
                                model.walkingtogether_err320: mean_mean_errors[7]
                                if FLAGS.seq_length_out >= 8
                                else None,
                                model.walkingtogether_err400: mean_mean_errors[9]
                                if FLAGS.seq_length_out >= 10
                                else None,
                                model.walkingtogether_err560: mean_mean_errors[13]
                                if FLAGS.seq_length_out >= 14
                                else None,
                                model.walkingtogether_err1000: mean_mean_errors[24]
                                if FLAGS.seq_length_out >= 25
                                else None,
                            },
                        )

                    for i in np.arange(len(summaries)):
                        model.test_writer.add_summary(summaries[i], current_step)

                print(
                    "============================\n"
                    "Global step:         %d\n"
                    "Learning rate:       %.4f\n"
                    "Step-time (ms):     %.4f\n"
                    "Train loss avg:      %.4f\n"
                    "--------------------------\n"
                    "Val loss:            %.4f\n"
                    "srnn loss:           %.4f\n"
                    "============================"
                    % (
                        model.global_step.eval(),
                        model.learning_rate.eval(),
                        step_time * 1000,
                        loss,
                        val_loss,
                        srnn_loss,
                    )
                )

                previous_losses.append(loss)

                # Calculate average mean error
                avg_mean_errors /= len(actions)
                avg_mean_errors_sum = sum(avg_mean_errors[range(FLAGS.seq_length_out)])
                summaries = sess.run(
                    [
                        model.average_err80_summary,
                        model.average_err160_summary,
                        model.average_err320_summary,
                        model.average_err400_summary,
                        model.average_err560_summary,
                        model.average_err1000_summary,
                        model.average_errsum_summary,
                    ],
                    {
                        model.average_err80: avg_mean_errors[1]
                        if FLAGS.seq_length_out >= 2
                        else None,
                        model.average_err160: avg_mean_errors[3]
                        if FLAGS.seq_length_out >= 4
                        else None,
                        model.average_err320: avg_mean_errors[7]
                        if FLAGS.seq_length_out >= 8
                        else None,
                        model.average_err400: avg_mean_errors[9]
                        if FLAGS.seq_length_out >= 10
                        else None,
                        model.average_err560: avg_mean_errors[13]
                        if FLAGS.seq_length_out >= 14
                        else None,
                        model.average_err1000: avg_mean_errors[24]
                        if FLAGS.seq_length_out >= 25
                        else None,
                        model.average_errsum: avg_mean_errors_sum
                        if FLAGS.seq_length_out >= 10
                        else None,
                    },
                )
                for i in np.arange(len(summaries)):
                    model.test_writer.add_summary(summaries[i], current_step)
                if (
                    avg_mean_errors[9] <= best_avg_mean_errors[-1]
                    or avg_mean_errors_sum <= best_avg_mean_errors_sum
                ):
                    print("Found a better model")
                    best_avg_mean_errors_sum = avg_mean_errors_sum
                    best_avg_mean_errors = avg_mean_errors[[1, 3, 7, 9]]
                    best_current_step = current_step
                    print("Best model's checkpoint id: {0}".format(best_current_step))
                    print(
                        "Best model's avg_mean_errors_sum: {0}".format(
                            best_avg_mean_errors_sum
                        )
                    )
                    print(
                        "Best model's avg_mean_errors: {0}".format(best_avg_mean_errors)
                    )

                # Save the model
                if (
                    current_step % FLAGS.save_every == 0  #1000
                    and current_step == best_current_step
                ):
                    print("Saving the model...")
                    start_time = time.time()
                    print("at {0}".format(train_dir))
                    model.saver.save(
                        sess,
                        os.path.normpath(os.path.join(train_dir, "checkpoint")),
                        global_step=current_step,
                    )
                    print(
                        "done in {0:.2f} ms".format((time.time() - start_time) * 1000)
                    )
                    print("Best model's checkpoint id: {0}".format(best_current_step))
                    print(
                        "Best model's avg_mean_errors_sum: {0}".format(
                            best_avg_mean_errors_sum
                        )
                    )
                    print(
                        "Best model's avg_mean_errors: {0}".format(best_avg_mean_errors)
                    )

                    # mox.file.copy_parallel(model_dir, FLAGS.train_url)

                # Reset global time and loss
                step_time, loss = 0, 0

                sys.stdout.flush()
        print("Saving the last model...")
        start_time = time.time()
        print("at {0}".format(train_dir))
        model.saver.save(
            sess,
            os.path.normpath(os.path.join(train_dir, "checkpoint")),
            global_step=current_step,
        )
        print("done in {0:.2f} ms".format((time.time() - start_time) * 1000))
        print("Best model's checkpoint id: {0}".format(best_current_step))
        print("Best model's avg_mean_errors_sum: {0}".format(best_avg_mean_errors_sum))
        print("Best model's avg_mean_errors: {0}".format(best_avg_mean_errors))


def get_srnn_gts(
    actions, model, test_set, data_mean, data_std, dim_to_ignore, one_hot, to_euler=True
):
    """
    Get the ground truths for srnn's sequences, and convert to Euler angles.
    (the error is always computed in Euler angles).

    Args
      actions: a list of actions to get ground truths for.
      model: training model we are using (we only use the "get_batch" method).
      test_set: dictionary with normalized training data.
      data_mean: d-long vector with the mean of the training data.
      data_std: d-long vector with the standard deviation of the training data.
      dim_to_ignore: dimensions that we are not using to train/predict.
      one_hot: whether the data comes with one-hot encoding indicating action.
      to_euler: whether to convert the angles to Euler format or keep thm in exponential map

    Returns
      srnn_gts_euler: a dictionary where the keys are actions, and the values
        are the ground_truth, denormalized expected outputs of srnns's seeds.
    """
    """
     获取 srnn 序列的基本事实，并转换为欧拉角。
     （误差总是以欧拉角计算）。

     参数
       动作：获取基本事实的动作列表。
       模型：我们正在使用的训练模型（我们只使用“get_batch”方法）。
       test_set：具有标准化训练数据的字典。
       data_mean：d-long 向量，训练数据的均值。
       data_std：d-long 向量，带有训练数据的标准差。
       dim_to_ignore：我们不用于训练/预测的维度。
       one_hot：数据是否带有one-hot编码指示动作。
       to_euler: 是否将角度转换为欧拉格式或将 thm 保留在指数图中

     退货
       srnn_gts_euler：一个字典，其中键是动作，值是
         是ground_truth，srnns 种子的非规范化预期输出。
  """
    srnn_gts_euler = {}

    for action in actions:

        srnn_gt_euler = []
        (
            _,
            _,
            srnn_expmap,
        ) = model.get_batch_srnn(test_set, action, actions)

        # expmap -> rotmat -> euler
        for i in np.arange(srnn_expmap.shape[0]):
            denormed = data_utils.unNormalizeData(
                srnn_expmap[i, :, :],
                data_mean,
                data_std,
                dim_to_ignore,
                actions,
                one_hot,
            )

            if to_euler:
                for j in np.arange(denormed.shape[0]):
                    for k in np.arange(3, 97, 3):
                        denormed[j, k: k + 3] = data_utils.rotmat2euler(
                            data_utils.expmap2rotmat(denormed[j, k: k + 3])
                        )

            srnn_gt_euler.append(denormed)

        # Put back in the dictionary
        srnn_gts_euler[action] = srnn_gt_euler

    return srnn_gts_euler


def sample():
    """Sample predictions for srnn's seeds"""
    print("===========this is sample==============")

    if FLAGS.load <= 0:
        raise (ValueError, "Must give an iteration to read parameters from")

    actions = define_actions(FLAGS.action)

    # Use the CPU if asked to
    # device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
    with tf.Session(config=npu_config_proto(config_proto=tf.ConfigProto())) as sess:
        # with tf.Session(config=npu_config_proto(config_proto=tf.ConfigProto( device_count = device_count ))) as sess:

        # === Create the model ===
        sampling = True
        model = create_model(sess, sampling)
        print("Model created")

        # Load all the data
        (
            train_set,
            test_set,
            data_mean,
            data_std,
            dim_to_ignore,
            dim_to_use,
        ) = read_all_data(
            actions,
            FLAGS.seq_length_in,
            FLAGS.seq_length_out,
            FLAGS.locals_dir,                           #--------------add
            not FLAGS.omit_one_hot,
        )

        # === Read and denormalize the gt with srnn's seeds, as we'll need them
        # many times for evaluation in Euler Angles ===
        srnn_gts_expmap = get_srnn_gts(
            actions,
            model,
            test_set,
            data_mean,
            data_std,
            dim_to_ignore,
            not FLAGS.omit_one_hot,
            to_euler=False,
        )
        srnn_gts_euler = get_srnn_gts(
            actions,
            model,
            test_set,
            data_mean,
            data_std,
            dim_to_ignore,
            not FLAGS.omit_one_hot,
        )

        # Clean and create a new h5 file of samples
        SAMPLES_FNAME = "samples.h5"
        try:
            os.remove(SAMPLES_FNAME)
        except OSError:
            pass

        avg_mean_errors = np.zeros([model.target_seq_len])
        # Predict and save for each action
        for action in actions:

            # Make prediction with srnn' seeds
            encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn(
                test_set, action, actions
            )
            forward_only = True
            srnn_seeds = True
            srnn_loss, srnn_poses, _, = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                decoder_outputs,
                forward_only,
                srnn_seeds,
            )

            # denormalizes too
            srnn_pred_expmap = data_utils.revert_output_format(
                srnn_poses,
                data_mean,
                data_std,
                dim_to_ignore,
                actions,
                not FLAGS.omit_one_hot,
            )

            # Save the conditioning seeds

            # Save the samples
            with h5py.File(SAMPLES_FNAME, "a") as hf:
                for i in np.arange(8):
                    # Save conditioning ground truth
                    node_name = "expmap/gt/{1}_{0}".format(i, action)
                    hf.create_dataset(node_name, data=srnn_gts_expmap[action][i])
                    # Save prediction
                    node_name = "expmap/preds/{1}_{0}".format(i, action)
                    hf.create_dataset(node_name, data=srnn_pred_expmap[i])

            # Compute and save the errors here
            mean_errors = np.zeros(
                (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0])
            )

            for i in np.arange(8):

                eulerchannels_pred = srnn_pred_expmap[i]

                for j in np.arange(eulerchannels_pred.shape[0]):
                    for k in np.arange(3, 97, 3):
                        eulerchannels_pred[j, k: k + 3] = data_utils.rotmat2euler(
                            data_utils.expmap2rotmat(eulerchannels_pred[j, k: k + 3])
                        )

                eulerchannels_pred[:, 0:6] = 0

                # Pick only the dimensions with sufficient standard deviation. Others are ignored.
                idx_to_use = np.where(np.std(eulerchannels_pred, 0) > 1e-4)[0]

                euc_error = np.power(
                    srnn_gts_euler[action][i][:, idx_to_use]
                    - eulerchannels_pred[:, idx_to_use],
                    2,
                )
                euc_error = np.sum(euc_error, 1)
                euc_error = np.sqrt(euc_error)
                print("sequence: ", i, "error: ", euc_error)
                mean_errors[i, :] = euc_error

            mean_mean_errors = np.mean(mean_errors, 0)
            print(action)
            print(",".join(map(str, mean_mean_errors.tolist())))

            print("Subset results for 80ms, 160ms, 320ms, 400ms")
            print(",".join(map(str, mean_mean_errors[[1, 3, 7, 9]].tolist())))
            if FLAGS.seq_length_out >= 25:
                print("Subset results for 80ms, 160ms, 320ms, 400ms, 560ms, 1000ms")
                print(
                    ",".join(map(str, mean_mean_errors[[1, 3, 7, 9, 13, 24]].tolist()))
                )
            avg_mean_errors += mean_mean_errors

            with h5py.File(SAMPLES_FNAME, "a") as hf:
                node_name = "mean_{0}_error".format(action)
                hf.create_dataset(node_name, data=mean_mean_errors)

        avg_mean_errors /= len(actions)
        print("Avg mean errors accross all actions:")
        print(",".join(map(str, avg_mean_errors.tolist())))
        print("Avg mean errors accross all actions:")
        print(",".join(map(str, avg_mean_errors[[1, 3, 7, 9]].tolist())))
        if FLAGS.seq_length_out >= 25:
            print("Subset results for 80ms, 160ms, 320ms, 400ms, 560ms, 1000ms")
            print(",".join(map(str, avg_mean_errors[[1, 3, 7, 9, 13, 24]].tolist())))

    return


def define_actions(action):
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """

    actions = [
        "walking",
        "eating",
        "smoking",
        "discussion",
        "directions",
        "greeting",
        "phoning",
        "posing",
        "purchases",
        "sitting",
        "sittingdown",
        "takingphoto",
        "waiting",
        "walkingdog",
        "walkingtogether",
    ]

    if action in actions:
        return [action]

    if action == "all":
        return actions

    if action == "all_srnn":
        return ["walking", "eating", "smoking", "discussion"]

    raise (ValueError, "Unrecognized action: %d" % action)


def read_all_data(actions, seq_length_in, seq_length_out, data_dir, one_hot):
    """
    Loads data for training/testing and normalizes it.

    Args
      actions: list of strings (actions) to load
      seq_length_in: number of frames to use in the burn-in sequence
      seq_length_out: number of frames to use in the output sequence
      data_dir: directory to load the data from
      one_hot: whether to use one-hot encoding per action
    Returns
      train_set: dictionary with normalized training data
      test_set: dictionary with test data
      data_mean: d-long vector with the mean of the training data
      data_std: d-long vector with the standard dev of the training data
      dim_to_ignore: dimensions that are not used becaused stdev is too small
      dim_to_use: dimensions that we are actually using in the model
    """

    # === Read training data ===
    print(
        "Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
            seq_length_in, seq_length_out
        )
    )

    train_subject_ids = [1, 6, 7, 8, 9, 11]
    test_subject_ids = [5]

    train_set, complete_train = data_utils.load_data(
        data_dir, train_subject_ids, actions, one_hot
    )
    test_set, complete_test = data_utils.load_data(
        data_dir, test_subject_ids, actions, one_hot
    )

    # Compute normalization stats
    datadic = data_utils.normalization_stats(
        complete_train
    )

    data_mean=datadic["data_mean"]
    data_std=datadic["data_std"]
    dimensions_to_ignore=datadic["dimensions_to_ignore"]          #add
    dimensions_to_use=datadic["dimensions_to_use"]                #add
    #Normalize -- subtract mean, divide by stdev
    train_set = data_utils.normalize_data(
        train_set, data_mean, data_std, dimensions_to_use, actions, one_hot
    )
    test_set = data_utils.normalize_data(
        test_set, data_mean, data_std, dimensions_to_use, actions, one_hot
    )
    print("done reading data.")

    return train_set, test_set, data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def main(_):
    os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "2"
    # os.environ['DUMP_GE_GRAPH'] = '2'
    # os.environ["DUMP_GRAPH_PATH"] = dump_graph_dir  # 指定dump图生成路径
    if FLAGS.sample:
        print("sample")
        sample()
    else:
        print("train is start")
        train()
        print("train is over")
        #mox.file.copy_parallel(train_dir, FLAGS.train_url)
        print("copy is over111")
        #mox.file.copy_parallel("/var/log/npu", FLAGS.train_url)
        print("copy is over222")


if __name__ == "__main__":
    npu_keras_sess = set_keras_session_npu_config()
    tf.app.run()
    close_session(npu_keras_sess)
