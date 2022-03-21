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
from npu_bridge.npu_init import *
import os
import tensorflow as tf
import numpy as np
from src.flownet2 import FlowNet2
import datetime
from src.data_base import Sintel
from src.flowlib import flow_to_image
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
flags = tf.flags
FLAGS = flags.FLAGS
## Required parameters
flags.DEFINE_string("result", "./log/flownet2", "The result directory where the model checkpoints will be written.")
flags.DEFINE_string("test_file", "./data/sintel/val.txt", "test dataset file path")
flags.DEFINE_string("dataset", "./data/sintel", "dataset path")
flags.DEFINE_string("checkpoint", './checkpoints/FlowNet2/flownet-2.ckpt-0', "checkpoint path")
#flags.DEFINE_string("checkpoint", './log/flownet2/model1069-loss-2.117536.ckpt', "checkpoint path")

## Other parametersresult
flags.DEFINE_list("image_size", [436, 1024], "size of input images")
flags.DEFINE_integer("batch_size", 1, "batch size for one GPU")
flags.DEFINE_float("weight_decay", 0.0004, "weight decay rate used in optimizer")
flags.DEFINE_string("chip", "npu", "Run on which chip, (npu or gpu or cpu)")
flags.DEFINE_string("gpu_device", '0,1', "gpu devices used in training")

if FLAGS.chip.lower() == 'gpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device
elif FLAGS.chip.lower() == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("**********")
    print("===>>>dataset:{}".format(FLAGS.dataset))
    print("===>>>result:{}".format(FLAGS.result))

    training_schedule = {
        'weight_decay': FLAGS.weight_decay,
    }
    w, h = FLAGS.image_size
    # Graph construction
    im1_pl = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, w, h, 3])
    im2_pl = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, w, h, 3])
    flow = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, w, h, 2])
    data = Sintel(batch_size=FLAGS.batch_size, base_path=FLAGS.dataset, img_size=FLAGS.image_size, val_data=FLAGS.test_file,status='clean',augment=False)
                
    flownet2 = FlowNet2()
    inputs = {'input_a': im1_pl, 'input_b': im2_pl}
    predictions = flownet2.model(inputs, training_schedule, trainable=False)
    total_loss = flownet2.loss(flow, predictions, regular_loss=False)
    #   pred_flow = flow_dict['flow']

    if FLAGS.chip.lower() == 'npu':
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map['use_off_line'].b = True
        #custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
        #custom_op.parameter_map["enable_dump"].b = True
        #custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes('./dump')
        #custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0")
        #custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
        custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes('allow_mix_precision')
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
    else:
        # config = make_config(FLAGS)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        #init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        init_op = tf.group(tf.global_variables_initializer())
        sess.run(init_op)
       
        saver = tf.train.Saver(tf.global_variables())
        if FLAGS.checkpoint:
            tf.logging.info('Loading checkpoint {}...'.format(FLAGS.checkpoint))
            # model_file=tf.train.latest_checkpoint(checkpoints)
            saver.restore(sess, FLAGS.checkpoint)

        val_next_batch = data.db_test.make_one_shot_iterator().get_next()
        val_batch_num = data.get_val_batch_num()
        tf.logging.info(" validation: batch_num=%d", val_batch_num)
        test_loss_list = []
               
        for batch_count in range(val_batch_num):
            start = datetime.datetime.now()
            val_input_a, val_input_b, val_flow = sess.run(val_next_batch)
            val_feed = {im1_pl:val_input_a, im2_pl:val_input_b, flow:val_flow}
            end = datetime.datetime.now()
            data_deltatime = (end - start).total_seconds() * 1000
            start = datetime.datetime.now()

            test_loss = sess.run([total_loss], feed_dict=val_feed)
            end = datetime.datetime.now()
            train_deltatime = (end - start).total_seconds() * 1000
            tf.logging.info("Time Used===>>>[FP+BP]:{:.3f} (ms), [Get Data]:{:.3f} (ms)".format(train_deltatime, data_deltatime))
            test_loss_list.append(test_loss)
        test_loss_list = np.array(test_loss_list)
        test_loss = np.mean(test_loss_list)
        tf.logging.info("test_loss = %s", test_loss/(int(FLAGS.image_size[0])*int(FLAGS.image_size[1])))


if __name__ == '__main__':
    flags.mark_flag_as_required("dataset")
    flags.mark_flag_as_required("result")
    flags.mark_flag_as_required("test_file")
    flags.mark_flag_as_required("checkpoint")
    main()

