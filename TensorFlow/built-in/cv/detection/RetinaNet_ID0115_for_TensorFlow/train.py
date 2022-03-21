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
flags = tf.app.flags
flags.DEFINE_string(name='data_path', default="",
                     help='the path of train data')
flags.DEFINE_string(name='ckpt_path', default="",
                     help='the resnet_ckpt for train') 
flags.DEFINE_string(name='model_dir', default="",
                     help='the path of save ckpt') 
flags.DEFINE_integer(name='steps', default="100000",
                     help='the step of train')
flags.DEFINE_string(name='precision_mode', default= 'allow_fp32_to_fp16',
                    help='allow_fp32_to_fp16/force_fp16/force_fp32 ' 
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
flags.DEFINE_boolean(name='autotune', default=False,
                    help='whether to enable autotune, default is False')  
FLAGS = flags.FLAGS

from networks import backbone
import numpy as np
import time
import queue
import threading
from npu_bridge.npu_init import *
from utils import generate_anchors, read_batch_data
from ops import smooth_l1, focal_loss
from config import BATCH_SIZE, IMG_H, IMG_W, K, WEIGHT_DECAY, LEARNING_RATE

anchors_p3 = generate_anchors(area=32, stride=8)
anchors_p4 = generate_anchors(area=64, stride=16)
anchors_p5 = generate_anchors(area=128, stride=32)
anchors_p6 = generate_anchors(area=256, stride=64)
anchors_p7 = generate_anchors(area=512, stride=128)
anchors = np.concatenate((anchors_p3, anchors_p4, anchors_p5, anchors_p6, anchors_p7), axis=0)

def npu_tf_optimizer(opt):
    npu_opt = NPUDistributedOptimizer(opt)
    return npu_opt

def queue_thread():
    for i in range(FLAGS.steps):
         (IMGS, FOREGROUND_MASKS, VALID_MASKS, LABELS, TARGET_BBOXES) = read_batch_data(anchors)
         q.put((IMGS, FOREGROUND_MASKS, VALID_MASKS, LABELS, TARGET_BBOXES))
         #print(q.qsize())

q = queue.Queue(10)
t = threading.Thread(target=queue_thread, args=())

def train():
    inputs = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_H, IMG_W, 3])
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, None, K])
    target_bbox = tf.placeholder(tf.float32, [BATCH_SIZE, None, 4])
    foreground_mask = tf.placeholder(tf.float32, [BATCH_SIZE, None])
    valid_mask = tf.placeholder(tf.float32, [BATCH_SIZE, None])
    is_training = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32)
    (class_logits, box_logits, _, _) = backbone(inputs, is_training)
    class_loss = (tf.reduce_sum((focal_loss(class_logits, labels) * valid_mask)) / tf.reduce_sum(foreground_mask))
    box_loss = (tf.reduce_sum((smooth_l1((box_logits - target_bbox)) * foreground_mask)) / tf.reduce_sum(foreground_mask))
    l2_reg = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    total_loss = ((class_loss + box_loss) + (l2_reg * WEIGHT_DECAY))

    #modify for npu start
    npu_int = npu_ops.initialize_system()
    npu_shutdown = npu_ops.shutdown_system()
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

    if FLAGS.autotune:
        custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")

    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess = tf.Session(config=config)
    sess.run(npu_int)
    with tf.variable_scope('opt'):
        Opt = npu_tf_optimizer(tf.train.MomentumOptimizer(learning_rate*get_rank_size(), momentum=0.9)).minimize(total_loss)
    input = tf.trainable_variables()
    bcast_global_variables_op = hccl_ops.broadcast(input, 0)
    checkpoint_dir = FLAGS.model_dir if get_rank_id()  == 0 else None
    #modify for npu end
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v2_50'))
    saver.restore(sess, FLAGS.ckpt_path + '/resnet_v2_50.ckpt')
    saver = tf.train.Saver()
    LR = LEARNING_RATE
    sess.run(bcast_global_variables_op)
    total_time = 0
    for i in range(FLAGS.steps):
        if (i == (60000/get_rank_size())):
            LR /= 10
        if (i == (80000/get_rank_size())):
            LR /= 10
        #(IMGS, FOREGROUND_MASKS, VALID_MASKS, LABELS, TARGET_BBOXES) = read_batch_data(anchors)
        queue_tuple = q.get()
        IMGS, FOREGROUND_MASKS, VALID_MASKS, LABELS, TARGET_BBOXES = queue_tuple[0], queue_tuple[1], queue_tuple[2], queue_tuple[3], queue_tuple[4]
        start_time = time.time()
        [_, TOTAL_LOSS, CLASS_LOSS, BOX_LOSS] = sess.run([Opt, total_loss, class_loss, box_loss], feed_dict={inputs: ((IMGS / 127.5) - 1.0), labels: LABELS, target_bbox: TARGET_BBOXES, foreground_mask: FOREGROUND_MASKS, valid_mask: VALID_MASKS, is_training: True, learning_rate: LR})
        total_time += (time.time() - start_time)
        if i % 100 == 0:
            #print(('Iteration: %d, Total Loss: %f, Class Loss: %f, Box Loss: %f' % (i, TOTAL_LOSS, CLASS_LOSS, BOX_LOSS)))
            print(('Iteration: %d, Total Loss: %f, Class Loss: %f, Box Loss: %f, total_time: %f, FLAGS.steps: %d' % (i, TOTAL_LOSS, CLASS_LOSS, BOX_LOSS,total_time,FLAGS.steps)))
        pass
    if checkpoint_dir:
        saver.save(sess, checkpoint_dir + '/model.ckpt')
    sess.run(npu_shutdown)
    sess.close()
    #print('Final Performance images/sec : %d' % (2/(total_time/FLAGS.steps)))
    print('Final Performance images/sec : %f' % (2/(total_time/FLAGS.steps)))

if (__name__ == '__main__'):
    t.start()
    train()
    t.join()
