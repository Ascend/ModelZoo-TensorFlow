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
#from math import nan
import os
import numpy as np
import datetime
from numpy.core.numeric import Inf

import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.client import timeline

from src.flownet2 import FlowNet2
from src.flowlib import flow_to_image
from src.data_base import Sintel
from npu_bridge.estimator.npu.npu_config import DumpConfig
flags = tf.flags
FLAGS = flags.FLAGS

#os.environ['DUMP_GE_GRAPH'] = '2'

## Required parameters
flags.DEFINE_string("result", "./log/flownet2", "The result directory where the model checkpoints will be written.")
flags.DEFINE_string("train_file", "./data/sintel/val.txt", "train dataset file path")
flags.DEFINE_string("val_file", "./data/sintel/val.txt", "val dataset file path")
flags.DEFINE_string("dataset", "./data/sintel", "dataset path")
flags.DEFINE_string("obs_dir", "obs://flownet", "obs result path, not need on gpu and apulis platform")

## Other parametersresult
#flags.DEFINE_list("image_size", [384, 768], "size of input images")
flags.DEFINE_list("image_size", [436, 1024], "size of input images")
flags.DEFINE_list("learning_rates", [1e-06, 5e-07, 2.5e-07, 1.25e-07, 6.25e-08, 1e-08], "The initial learning rate for Adam.")
flags.DEFINE_list("step_values", [8000, 8500, 9000, 9500, 10000], "The steps to change the learning rate for Adam.")
flags.DEFINE_float("momentum", 0.9, "The momentum used in Adam.")
flags.DEFINE_float("momentum2", 0.999, "The momentum2 used in Adam.")
flags.DEFINE_float("weight_decay", 0.0004, "weight decay rate used in optimizer")
flags.DEFINE_integer("batch_size", 4, "batch size for one GPU")
flags.DEFINE_integer("train_step", 10000, "total epochs for training")
flags.DEFINE_integer("save_step", 500, "epochs for saving checkpoint")
flags.DEFINE_integer("decay_step", 500, "update the learning_rate value every decay_steps times")
flags.DEFINE_string("resume_path", './log/flownet2' , "resume checkpoint path")
flags.DEFINE_string("pretrained", './checkpoints/FlowNet2/flownet-2.ckpt-0', "checkpoint path")
flags.DEFINE_string("chip", "npu", "Run on which chip, (npu or gpu or cpu)")
flags.DEFINE_string("gpu_device", '0', "gpu devices used in training")
flags.DEFINE_string("platform", "linux", "Run on linux/apulis/modelarts platform. Modelarts Platform has some extra data copy operations")
flags.DEFINE_boolean("profiling", False, "profiling for performance or not")
#add
flags.DEFINE_integer("max_iter_less", 0, "less train")
flags.DEFINE_integer("batch_num_less", 0, "less batch_num")

if FLAGS.chip.lower() == 'gpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device
elif FLAGS.chip.lower() == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("**********")
    print("===>>>dataset:{}".format(FLAGS.dataset))
    print("===>>>result:{}".format(FLAGS.result))
    print("===>>>train_step:{}".format(FLAGS.train_step))

    # print all parameters
    for attr, flag_obj in sorted(FLAGS.__flags.items()):
        print("{} = {}".format(attr.lower(), flag_obj.value))
    training_schedule = {
        'step_values': FLAGS.step_values,
        'learning_rates': FLAGS.learning_rates,
        'momentum': FLAGS.momentum,
        'momentum2': FLAGS.momentum2,
        'weight_decay': FLAGS.weight_decay,
        'max_iter': FLAGS.train_step,
    }

    ##============Obtain Data================
    model = FlowNet2()
    global_step = model.global_step
    w, h = FLAGS.image_size
    # input_a, input_b, flow = load_batch(FLYING_CHAIRS_DATASET_CONFIG, 'sample', global_step)
    input_a = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, w, h, 3])
    input_b = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, w, h, 3])
    flow = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, w, h, 2])
    
    data = Sintel(batch_size=FLAGS.batch_size, base_path=FLAGS.dataset, img_size=FLAGS.image_size,
                train_data=FLAGS.train_file,val_data=FLAGS.val_file,status='clean',augment=False)

    tf.summary.image("image_a", input_a, max_outputs=2)
    tf.summary.image("image_b", input_b, max_outputs=2)
    
    inputs = {
            'input_a': input_a,
            'input_b': input_b,
        }
        
    # construction model
    predictions = model.model(inputs, training_schedule)

    # define loss function and optimizer
    learning_rate = tf.train.piecewise_constant(
            global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])

    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        training_schedule['momentum'],
        training_schedule['momentum2'])
    total_loss = model.loss(flow, predictions)

    tf.summary.scalar('loss', total_loss)
    if 'flow' in predictions:
        pred_flow_0 = predictions['flow'][0, :, :, :]
        pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
        pred_flow_1 = predictions['flow'][1, :, :, :]
        pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
        pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
        tf.summary.image('pred_flow', pred_flow_img, max_outputs=2)

    true_flow_0 = flow[0, :, :, :]
    true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
    true_flow_1 = flow[1, :, :, :]
    true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
    true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
    tf.summary.image('true_flow', true_flow_img, max_outputs=2)
    summary_op = tf.summary.merge_all()

    ## gpu profiling configuration
    if FLAGS.chip.lower() == 'gpu' and FLAGS.profiling:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        options = None
        run_metadata = None

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
        if FLAGS.profiling:
            if not os.path.exists('./log/profiling'):
                os.mkdir('./log/profiling')
            custom_op.parameter_map['profiling_mode'].b = True
            #custom_op.parameter_map['profiling_options'].s = tf.compat.as_bytes('{ "output":"./log/profiling",\
            #             "training_trace":"on",\
            #             "task_trace":"on",\
            #             "aicpu":"on",\
            #             "fp_point":"FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/conv1/weights/Initializer/truncated_normal/TruncatedNormal",\
            #             "bp_point":"gradients/AddN_21" }')
            custom_op.parameter_map['profiling_options'].s = tf.compat.as_bytes('{ "output":"./log/profiling",\
                         "task_trace":"on",\
                         "training_trace":"on",\
                         "aicpu":"on",\
                         "fp_point":"FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/Pad",\
                         "bp_point":"gradients/AddN_21" }')
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    else:
        # config = make_config(FLAGS)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    min_loss = Inf
    # start training

    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)

        train_writer = tf.summary.FileWriter(logdir=os.path.join(FLAGS.result, "train"), graph=sess.graph)
        test_writer = tf.summary.FileWriter(logdir=os.path.join(FLAGS.result, "val"), graph=sess.graph)

        # saver is used to save the model
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        if FLAGS.pretrained:
            tf.logging.info('Loading checkpoint {}...'.format(FLAGS.pretrained))
            # model_file=tf.train.latest_checkpoint(checkpoints)
            saver.restore(sess, FLAGS.pretrained)
        # if FLAGS.resume_path and FLAGS.resume_path != '':#add
        #     tf.logging.info('Loading checkpoint from {}...'.format(FLAGS.resume_path))
        #     model_file=tf.train.latest_checkpoint(FLAGS.resume_path)
        #     saver.restore(sess, model_file)

        train = optimizer.minimize(total_loss,global_step=global_step)
        sess.run(tf.variables_initializer(optimizer.variables()))
        train_next_batch = data.db_train.make_one_shot_iterator().get_next()
        val_next_batch = data.db_test.make_one_shot_iterator().get_next()
        for step in range(training_schedule['max_iter']-FLAGS.max_iter_less): #add (10000)-9995
            batch_num = data.get_train_batch_num()
            tf.logging.info(" epoch = %d, batch_num=%d", step, batch_num)
            train_loss_list = []
            for batch_count in range(batch_num-FLAGS.batch_num_less): #add  (69)-50
                start = datetime.datetime.now()
                cur_input_a, cur_input_b, cur_flow = sess.run(train_next_batch)
                end = datetime.datetime.now()
                data_deltatime = (end - start).total_seconds() * 1000

                start = datetime.datetime.now()
                
                feed_dict = {input_a:cur_input_a, input_b:cur_input_b, flow:cur_flow}
                #_, train_loss, summary = sess.run([train, total_loss, summary_op], 
                _, train_loss = sess.run([train, total_loss], feed_dict=feed_dict,options=options,run_metadata=run_metadata)
                end = datetime.datetime.now()
                train_deltatime = (end - start).total_seconds() * 1000
                if batch_count % 20 == 0:
                    tf.logging.info("Time Used===>>>[FP+BP]:{:.3f} (ms) [data]:{:.3f} (ms)".format(train_deltatime, data_deltatime))
                train_loss_list.append(train_loss)
                
            train_loss = np.mean(train_loss_list)
            #train_writer.add_summary(summary, step)
            tf.logging.info("train_loss = %s", train_loss/(int(FLAGS.image_size[0])*int(FLAGS.image_size[1])))
            # tf.logging.info("train_loss = %s", train_loss)
            #add
            if (step + 1) % FLAGS.save_step == 0: #500
                val_batch_num = data.get_val_batch_num()
                tf.logging.info(" validation: batch_num=%d", val_batch_num)
                test_loss_list = []
                for batch_count in range(val_batch_num):
                    val_input_a, val_input_b, val_flow = sess.run(val_next_batch)

                    val_feed = {input_a:val_input_a, input_b:val_input_b, flow:val_flow}
                    #test_loss, summary = sess.run([total_loss, summary_op], feed_dict=val_feed)
                    test_loss = sess.run([total_loss], feed_dict=val_feed)
                    test_loss_list.append(test_loss)
                test_loss_list = np.array(test_loss_list)
                test_loss = np.mean(test_loss_list)
                #test_writer.add_summary(summary, step)
                tf.logging.info("test_loss = %s", test_loss/(int(FLAGS.image_size[0])*int(FLAGS.image_size[1])))
                # tf.logging.info("test_loss = %s", test_loss)

                # save model
                if test_loss < min_loss:
                    saver.save(sess=sess, save_path=os.path.join(FLAGS.result, 'model{}-loss-{:4f}.ckpt'.format(step, test_loss/(int(FLAGS.image_size[0])*int(FLAGS.image_size[1])))))
                    #test_writer.add_summary(summary=summary, global_step=step)
                    min_loss = test_loss
                    tf.logging.info("saved checkpoints!")
        saver.save(sess=sess, save_path=os.path.join(FLAGS.resume_path,"flownet2-ckpt"))
        tf.io.write_graph(sess.graph, logdir='./ckpt', name='graph.pbtxt', as_text=True)
        train_writer.close()
        test_writer.close()

    if FLAGS.chip.lower() == 'gpu' and FLAGS.profiling:
        work_dir = os.getcwd()
        timeline_path = os.path.join(work_dir, 'timeline.ctf.json')
        with open(timeline_path, 'w') as trace_file:
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            trace_file.write(trace.generate_chrome_trace_format())

    if FLAGS.platform.lower() == 'modelarts':
        from help_modelarts import modelarts_result2obs
        modelarts_result2obs(FLAGS)

if __name__ == "__main__":
    flags.mark_flag_as_required("dataset")
    flags.mark_flag_as_required("result")
    flags.mark_flag_as_required("train_file")
    flags.mark_flag_as_required("val_file")
    tf.app.run()
