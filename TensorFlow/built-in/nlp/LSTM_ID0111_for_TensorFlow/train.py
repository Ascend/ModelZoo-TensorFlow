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
import time
import os
import argparse
import datetime
from random import randint
import tensorflow as tf
from tensorflow.python.framework import dtypes
from npu_bridge.estimator.npu.npu_dynamic_rnn import DynamicRNN
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.npu_init import *

flags = tf.app.flags
flags.DEFINE_string(name='data_path', default="",
                     help='the path of train data')
flags.DEFINE_string(name='model_dir', default="",
                     help='the path of save ckpt') 
flags.DEFINE_integer(name='steps', default="100000",
                     help='the step of train')
flags.DEFINE_float(name='learning_rate', default="0.001",
                     help='the lr of train')
flags.DEFINE_integer(name='batch_size', default="24",
                     help='the bs of train')
flags.DEFINE_string(name='precision_mode', default= 'allow_mix_precision',
                    help='allow_fp32_to_fp16/force_fp16/ ' 
                    'must_keep_origin_dtype/allow_mix_precision.')
flags.DEFINE_boolean(name='over_dump', default=False,
                    help='if or not over detection, default is False')
flags.DEFINE_boolean(name='data_dump_flag', default=False,
                    help='data dump flag, default is False')
flags.DEFINE_string(name='data_dump_step', default="10",
                    help='data dump step, default is 10')
flags.DEFINE_boolean(name='profiling', default=False,
                    help='if or not profiling for performance debug, default is False') 
flags.DEFINE_string(name='profiling_dump_path', default="/home/data",
                    help='the path to save profiling data')                                      
flags.DEFINE_string(name='over_dump_path', default="/home/data",
                    help='the path to save over dump data')  
flags.DEFINE_string(name='data_dump_path', default="/home/data",
                    help='the path to save dump data')    
flags.DEFINE_integer(name='NPU_NUMS', default="1", help='the nums of NPU ')
FLAGS = flags.FLAGS

wordVectors = np.load(FLAGS.data_path + '/' + 'wordVectors.npy')
print ('Loaded the word vectors!')
wordVectors = np.pad(wordVectors, ((0,0),(0,14)))
print(wordVectors.shape)
ids = np.load(FLAGS.data_path + '/' + 'idsMatrix.npy')
print ('Loaded the idsMatrix.npy!')

def broadcast_global_variables(root_rank, index):
    op_list = []
    for var in tf.global_variables():
        if "float" in var.dtype.name:
            inputs = [var]
            outputs = hccl_ops.broadcast(tensor=inputs, root_rank=root_rank)
            if outputs is not None:
                op_list.append(outputs[0].op)
                op_list.append(tf.assign(var, outputs[0]))
    return tf.group(op_list)

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels

batchSize = FLAGS.batch_size
maxSeqLength = 250
lstmUnits = 64
numClasses = 2
numDimensions = 50
iterations = FLAGS.steps

tf.reset_default_graph()
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.cast(data, tf.float16)
wordVectors = tf.cast(wordVectors, tf.float16)
data = tf.nn.embedding_lookup(wordVectors,input_data)
data = tf.transpose(data, [1, 0, 2], name='transpose_time_major')
data = tf.strided_slice(data, [0, 0, 0], [maxSeqLength, batchSize, numDimensions])
data = tf.cast(data, tf.float32)
lstm = DynamicRNN(hidden_size=lstmUnits, dtype=tf.float32, forget_bias=1.0)
value, output_h, output_c, i, j, f, o, tanhct = lstm(data)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
last = tf.gather(value, (int(value.get_shape()[0]) - 1))
prediction = (tf.matmul(last, weight) + bias)
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)#.minimize(loss, global_step=global_step)
if FLAGS.NPU_NUMS == 8:
    optimizer = NPUDistributedOptimizer(optimizer)
loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager).minimize(loss)

#sess = tf.InteractiveSession()
#tf.summary.scalar('Loss', loss)
#tf.summary.scalar('Accuracy', accuracy)
#merged = tf.summary.merge_all()
#logdir = (('tensorboard/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')) + '/')
#writer = tf.summary.FileWriter(logdir, sess.graph)

#npu_config
config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(FLAGS.precision_mode)

if FLAGS.data_dump_flag:
    custom_op.parameter_map["enable_dump"].b = True
    custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(FLAGS.data_dump_path)
    custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(FLAGS.data_dump_step)
    custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

if FLAGS.over_dump:
    custom_op.parameter_map["enable_dump_debug"].b = True
    custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(FLAGS.over_dump_path)
    custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")

if FLAGS.profiling:
    custom_op.parameter_map["profiling_mode"].b = False
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"%s",\
                    "training_trace":"on",\
                    "task_trace":"on",\
                    "aicpu":"on",\
                    "fp_point":"",\
                    "bp_point":"",\
                    "aic_metrics":"PipeUtilization"}' % FLAGS.profiling_dump_path)

config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

bcast_op = broadcast_global_variables(0, 1)

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for i in range(iterations):
       #Next Batch of reviews
        (nextBatch, nextBatchLabels) = getTrainBatch()
        #start_time = time.time()
        if FLAGS.NPU_NUMS == 8:
            sess.run(bcast_op)
        start_time = time.perf_counter()
        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
        end_time = time.perf_counter()
       #Write summary to Tensorboard
        if ((i+1) % 100 == 0):
            #end_time = time.time()
            #summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            #writer.add_summary(summary, i)
            (l, acc) = sess.run([loss, accuracy], {input_data: nextBatch, labels: nextBatchLabels})
            print(('training! Step: %d Training loss: %f Training acc: %f Training fps: %d' % (i+1, l, acc,  (batchSize/(end_time-start_time)))), flush=True)
        
       #Save the network every 10,000 training iterations
        if ((i+1) % 10000 == 0 and i != 0):
            save_path = saver.save(sess, FLAGS.model_dir + '/pretrained_lstm.ckpt', global_step=i+1)
            print(('saved to %s' % save_path), flush=True)

    #writer.close()
    