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
#==============================================================================

from __future__ import print_function
from npu_bridge.npu_init import *
from tqdm import tqdm
import argparse
import os
import time
from data_load import make_dataset, get_batch, load_vocab
from hyperparams import Hyperparams as hp
from networks import encoder, decoder, converter
import tensorflow as tf
from fn_utils import infolog
import synthesize
from utils import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Graph():

    def __init__(self, input_data, config=None, training=True, train_form='Both'):
        (self.char2idx, self.idx2char) = load_vocab()
        # self.graph = tf.Graph()
        # with self.graph.as_default():
        if training:
            # (self.origx, self.x, self.y1, self.y2, self.y3, self.num_batch) = get_batch(config, train_form)
            # text_tests
            self.origx = input_data["origx"]
            # texts
            self.x = input_data["x"]
            # mels
            self.y1 = input_data["y1"]
            # dones
            self.y2 = None
            if "y2" in input_data.keys():
                self.y2 = input_data["y2"]
            # mags
            self.y3 = None
            if "y3" in input_data.keys():
                self.y3 = input_data["y3"]
            print("========After get_batch")
        else:
            self.x = tf.placeholder(tf.int32, shape=(config.batch_size, hp.T_x))
            self.y1 = tf.placeholder(tf.float32, shape=(config.batch_size, (hp.T_y // hp.r), (hp.n_mels * hp.r)))
        if (train_form != 'Converter'):
            with tf.variable_scope('encoder'):
                self.encoded = encoder(self.x, training=training)
            with tf.variable_scope('decoder'):
                (self.mel_output, self.bef_mel_output, self.done_output, self.decoder_state, self.LTSM, self.step) = \
                    decoder(self.y1, self.encoded, config.batch_size, training=training)
                self.cell_state = self.decoder_state.cell_state
                self.mel_output = tf.nn.sigmoid(self.mel_output)
        if (train_form == 'Both'):
            with tf.variable_scope('converter'):
                self.converter_input = self.mel_output
                self.mag_logits = converter(self.converter_input, training=training)
                self.mag_output = tf.nn.sigmoid(self.mag_logits)
        elif (train_form == 'Converter'):
            with tf.variable_scope('converter'):
                self.converter_input = self.y1
                self.mag_logits = converter(self.converter_input, training=training)
                self.mag_output = tf.nn.sigmoid(self.mag_logits)
                print("========After mag_logits converter")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if training:
            if (train_form != 'Converter'):
                self.loss1 = tf.reduce_mean(tf.abs((self.mel_output - self.y1)))
                self.loss1b = tf.reduce_mean(tf.abs((self.bef_mel_output - self.y1)))
                if hp.include_dones:
                    self.loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.done_output,
                                                                                               labels=self.y2))
            if (train_form != 'Encoder'):
                self.loss3 = tf.reduce_mean(tf.abs((self.mag_output - self.y3)))
                print("========After self.loss3")
            if (train_form == 'Both'):
                if hp.include_dones:
                    self.loss = (((self.loss1 + self.loss1b) + self.loss2) + self.loss3)
                else:
                    self.loss = ((self.loss1 + self.loss1b) + self.loss3)
            elif (train_form == 'Encoder'):
                if hp.include_dones:
                    self.loss = ((self.loss1 + self.loss1b) + self.loss2)
                else:
                    self.loss = (self.loss1 + self.loss1b)
            else:
                self.loss = self.loss3
                print("========After self.loss")

            #self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
            #self.optimizer = NPUDistributedOptimizer(self.optimizer)
            opt_tmp = tf.train.AdamOptimizer(learning_rate=hp.lr)
            # loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2*32,
            #                                                        incr_every_n_steps=1000,
            #                                                        decr_every_n_nan_or_inf=2,
            #                                                        decr_ratio=0.5)
            # overflow also update for dynamic_bs
            loss_scale_manager = FixedLossScaleManager(loss_scale=1024, enable_overflow_check=False)
            self.optimizer = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)

            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            print("========After self.gvs")
            for (grad, var) in self.gvs:
                grad = (grad if (grad is None) else tf.clip_by_value(grad, ((- 1.0) * hp.max_grad_val), hp.max_grad_val))
                grad = (grad if (grad is None) else tf.clip_by_norm(grad, hp.max_grad_norm))
                self.clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)
            tf.summary.scalar('loss', self.loss)
            print("========After tf.summary.scalar")
            if (train_form != 'Converter'):
                tf.summary.histogram('mel_output', self.mel_output)
                tf.summary.histogram('mel_actual', self.y1)
                tf.summary.scalar('loss1', self.loss1)
                if hp.include_dones:
                    tf.summary.histogram('done_output', self.done_output)
                    tf.summary.histogram('done_actual', self.y2)
                    tf.summary.scalar('loss2', self.loss2)
            if (train_form != 'Encoder'):
                tf.summary.histogram('mag_output', self.mag_output)
                tf.summary.histogram('mag_actual', self.y3)
                tf.summary.scalar('loss3', self.loss3)
            self.merged = tf.summary.merge_all()
            print("========After self.merged")

class GraphTest():

    def __init__(self, config=None):
        (self.char2idx, self.idx2char) = load_vocab()
        self.graph = tf.Graph()
        with self.graph.as_default():
            (self.origx, _, _, _, _, _) = get_batch(config, 'Encoder')

def main():
    print()
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default=hp.logdir)
    parser.add_argument('--log_name', default=hp.logname)
    parser.add_argument('--sample_dir', default=hp.sampledir)
    parser.add_argument('--data_paths', default=hp.data)
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--load_converter', default=None)
    parser.add_argument('--deltree', default=False)
    parser.add_argument('--summary_interval', type=int, default=hp.summary_interval)
    parser.add_argument('--test_interval', type=int, default=hp.test_interval)
    parser.add_argument('--checkpoint_interval', type=int, default=hp.checkpoint_interval)
    parser.add_argument('--num_iterations', type=int, default=hp.num_iterations)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=hp.batch_size)

    parser.add_argument('--data_dump_flag', type=str, default='False')
    parser.add_argument('--over_dump', type=str, default='False')
    parser.add_argument('--profiling', type=str, default='False')
    parser.add_argument('--precision_mode', type=str, default='allow_fp32_to_fp16',
                        help='allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision.')
    parser.add_argument('--data_dump_path', type=str, default='/home/data')
    parser.add_argument('--data_dump_step', type=str, default='10')
    parser.add_argument('--over_dump_path', type=str, default='/home/data')
    parser.add_argument('--profiling_dump_path', type=str, default='/home/data')
    parser.add_argument('--dynamic_bs', default=False)

    config = parser.parse_args()
    config.log_dir = ((config.log_dir + '/') + config.log_name)
    if (not os.path.exists(config.log_dir)):
        os.makedirs(config.log_dir)
    elif config.deltree:
        for the_file in os.listdir(config.log_dir):
            file_path = os.path.join(config.log_dir, the_file)
            os.unlink(file_path)
    log_path = os.path.join((config.log_dir + '/'), 'train.log')
    infolog.init(log_path, 'log')
    checkpoint_path = os.path.join(config.log_dir, 'model.ckpt')

    session_config = tf.ConfigProto()
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(config.precision_mode)
    if config.dynamic_bs:
        # custom_op.parameter_map["dynamic_input"].b = True
        # custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("dynamic_execute")
        # custom_op.parameter_map["dynamic_inputs_shape_range"].s = tf.compat.as_bytes("data:[1~200,200],[1~200,810,80],[1~200,810,1025]")
        # add for bs=128/160
        custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(30 * 1024 * 1024 * 1024))
        custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(1 * 1024 * 1024 * 1024))
    else:
        custom_op.parameter_map["enable_data_pre_proc"].b = True

    if config.data_dump_flag == 'True':
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(config.data_dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(config.data_dump_step)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
    if config.over_dump == 'True':
        custom_op.parameter_map["enable_dump_debug"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(config.over_dump_path)
        custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
    if config.profiling == 'True':
        custom_op.parameter_map["profiling_mode"].b = True
        custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"' + config.profiling_dump_path + '", \
                                                                           "task_trace":"on", \
                                                                            "training_trace":"on", \
                                                                            "fp_point":"converter/converter/converter_rnn/add_1converter/converter/converter_rnn/mul_1", \
                                                                            "bp_point":"gradients/converter/converter/converter_conv/converter_conv_0/l2_normalize/Square_grad/Mulgradients/converter/converter/converter_conv/converter_conv_0/l2_normalize/Square_grad/Mul_1gradients/AddN_20"}')

    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.function_optimization = RewriterConfig.OFF
    npu_config = npu_config_proto(config_proto=session_config)

    train_dataset, num_batch = make_dataset(config, hp.train_form)
    iterator = train_dataset.make_initializable_iterator()

    if (hp.train_form == 'Both'):
        if hp.include_dones:
            (texts, texts_tests, mels, mags, dones) = iterator.get_next()
            input_data = {"origx": texts_tests, "x": texts, "y1": mels, "y2": dones, "y3": mags}
        else:
            (texts, texts_tests, mels, mags) = iterator.get_next()
            input_data = {"origx": texts_tests, "x": texts, "y1": mels, "y3": mags}
    elif (hp.train_form == 'Encoder'):
        if hp.include_dones:
            (texts, texts_tests, mels, dones) = iterator.get_next()
            input_data = {"origx": texts_tests, "x": texts, "y1": mels, "y2": dones}
        else:
            (texts, texts_tests, mels) = iterator.get_next()
            input_data = {"origx": texts_tests, "x": texts, "y1": mels}
    else:
        (texts, texts_tests, mels, mags) = iterator.get_next()
        input_data = {"origx": texts_tests, "x": texts, "y1": mels, "y3": mags}

    if (hp.test_only == 0):
        g = Graph(input_data=input_data, config=config, training=True, train_form=hp.train_form)
        print('Training Graph loaded')
    if (hp.test_graph or (hp.test_only > 0)):
        g2 = Graph(config=config, training=False, train_form=hp.train_form)
        print('Testing Graph loaded')
        if (config.load_converter or (hp.test_only > 0)):
            g_conv = Graph(config=config, training=False, train_form='Converter')
            print('Converter Graph loaded')
    print("========After graph")

    num_step = config.num_iterations if config.num_iterations < num_batch else num_batch
    print("========After session config")
    if (hp.test_only == 0):
        with tf.Session(config=npu_config) as sess:
            if config.load_path:
                infolog.log(('Resuming from checkpoint: %s ' % tf.train.latest_checkpoint(config.log_dir)), slack=True)
                tf.train.Saver().restore(sess, tf.train.latest_checkpoint(config.log_dir))
            else:
                infolog.log('Starting new training', slack=True)
            # summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)
            # print("========After summary_writer")
            sess.run(iterator.initializer)
            sess.run(tf.global_variables_initializer())
            for epoch in range(1, config.epoch):
                losses = [0, 0, 0, 0, 0]
                for step in tqdm(range(num_step)):
                    startTime = time.time()
                    if (hp.train_form == 'Both'):
                        if hp.include_dones:
                            fetch = [g.global_step, g.merged, g.loss, g.loss1, g.loss1b, g.loss2, g.loss3, g.train_op]
                            (gs, merged, loss, loss1, loss1b, loss2, loss3, _) = sess.run(fetches=fetch)
                            loss_one = [loss, loss1, loss1b, loss2, loss3]
                        else:
                            fetch = [g.global_step, g.loss, g.loss1, g.loss1b, g.loss3, g.train_op]
                            (gs, loss, loss1, loss1b, loss3, _) = sess.run(fetches=fetch)
                            loss_one = [loss, loss1, loss1b, loss3, 0]
                    elif (hp.train_form == 'Encoder'):
                        if hp.include_dones:
                            fetch = [g.global_step, g.merged, g.loss, g.loss1, g.loss1b, g.loss2, g.train_op]
                            (gs, merged, loss, loss1, loss1b, loss2, _) = sess.run(fetches=fetch)
                            loss_one = [loss, loss1, loss1b, loss2, 0]
                        else:
                            fetch = [g.global_step, g.loss, g.loss1, g.loss1b, g.train_op]
                            (gs, loss, loss1, loss1b, _) = sess.run(fetches=fetch)
                            loss_one = [loss, loss1, loss1b, 0, 0]
                    else:
                        print("========before sess.run")
                        fetch = [g.global_step, g.merged, g.loss, g.train_op]
                        (gs, merged, loss, _) = sess.run(fetches=fetch)
                        loss_one = [loss, 0, 0, 0, 0]
                        print("========After sess.run")
                    endTime = time.time()
                    losses = [(x + y) for (x, y) in zip(losses, loss_one)]
                    infolog.log(('Epoch %d (%04d): Loss = %.8f perf = %.4f' % (epoch, gs, loss, endTime - startTime)))
                # losses = [(x / g.num_batch) for x in losses]
                losses = [(x / num_batch) for x in losses]
                print('###############################################################################')
                if (hp.train_form == 'Both'):
                    if hp.include_dones:
                        infolog.log(('Epoch %d (%04d): Loss = %.8f Loss1 = %.8f Loss1b = %.8f Loss2 = %.8f Loss3 = %.8f' % (epoch, gs, losses[0], losses[1], losses[2], losses[3], losses[4])))
                    else:
                        infolog.log(('Epoch %d (%04d): Loss = %.8f Loss1 = %.8f Loss1b = %.8f Loss3 = %.8f' % (epoch, gs, losses[0], losses[1], losses[2], losses[3])))
                elif (hp.train_form == 'Encoder'):
                    if hp.include_dones:
                        infolog.log(('Epoch %d (%04d): Loss = %.8f Loss1 = %.8f Loss1b = %.8f Loss2 = %.8f' % (epoch, gs, losses[0], losses[1], losses[2], losses[3])))
                    else:
                        infolog.log(('Epoch %d (%04d): Loss = %.8f Loss1 = %.8f Loss1b = %.8f' % (epoch, gs, losses[0], losses[1], losses[2])))
                else:
                    pass
                print('###############################################################################')
                print("========After for num_batch")
                if not config.dynamic_bs and ((epoch % config.summary_interval) == 0):
                    infolog.log('Saving summary')
                    # summary_writer.add_summary(merged, gs)
                    if (hp.train_form == 'Both'):
                        if hp.include_dones:
                            (origx, Kmel_out, Ky1, Kdone_out, Ky2, Kmag_out, Ky3) = sess.run([g.origx, g.mel_output, g.y1, g.done_output, g.y2, g.mag_output, g.y3])
                            plot_losses(config, Kmel_out, Ky1, Kdone_out, Ky2, Kmag_out, Ky3, gs)
                        else:
                            (origx, Kmel_out, Ky1, Kmag_out, Ky3) = sess.run([g.origx, g.mel_output, g.y1, g.mag_output, g.y3])
                            plot_losses(config, Kmel_out, Ky1, None, None, Kmag_out, Ky3, gs)
                    elif (hp.train_form == 'Encoder'):
                        if hp.include_dones:
                            (origx, Kmel_out, Ky1, Kdone_out, Ky2) = sess.run([g.origx, g.mel_output, g.y1, g.done_output, g.y2])
                            plot_losses(config, Kmel_out, Ky1, Kdone_out, Ky2, None, None, gs)
                        else:
                            (origx, Kmel_out, Ky1) = sess.run([g.origx, g.mel_output, g.y1])
                            plot_losses(config, Kmel_out, Ky1, None, None, None, None, gs)
                    else:
                        (origx, Kmag_out, Ky3) = sess.run([g.origx, g.mag_output, g.y3])
                        plot_losses(config, None, None, None, None, Kmag_out, Ky3, gs)
                if ((epoch % config.checkpoint_interval) == 0):
                    tf.train.Saver().save(sess, checkpoint_path, global_step=gs)
                if (hp.test_graph and (hp.train_form != 'Converter')):
                    if ((epoch % config.test_interval) == 0):
                        infolog.log('Saving audio')
                        origx = sess.run([g.origx])
                        if (not config.load_converter):
                            wavs = synthesize.synthesize_part(g2, config, gs, origx, None)
                        else:
                            wavs = synthesize.synthesize_part(g2, config, gs, origx, g_conv)
                        plot_wavs(config, wavs, gs)
                if (gs >= config.num_iterations):
                    break
    else:
        infolog.log('Saving audio')
        gT = GraphTest(config=config)
        with gT.graph.as_default():
            svT = tf.train.Supervisor(logdir=config.log_dir)
            with svT.managed_session(config=session_config) as sessT:
                origx = sessT.run([gT.origx])
        if (not config.load_converter):
            wavs = synthesize.synthesize_part(g2, config, 0, origx, None)
        else:
            wavs = synthesize.synthesize_part(g2, config, 0, origx, g_conv)
        plot_wavs(config, wavs, 0)
    print('Done')
if (__name__ == '__main__'):
    main()
