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

from skimage.transform import resize
import argparse
import os
import pathlib
import time
import tensorflow as tf
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from callbacks import get_nyu_callbacks
from loss import depth_loss_function
from model import create_model
from utils import load_test_data
from data import create_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ENABLE_FORCE_V2_CONTROL'] = "1"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--bs', type=int, default=4, help='Batch size')
    parser.add_argument('--steps', type=int, default=12672, help='Steps per epoch')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--mindepth', type=float, default=10.0, help='Minimum of input depths')
    parser.add_argument('--maxdepth', type=float, default=1000.0, help='Maximum of input depths')
    parser.add_argument('--name', type=str, default='densedepth_nyu', help='A name to attach to the training session')
    parser.add_argument('--ckptdir', type=str, default='', help='Start training from an existing model.')
    parser.add_argument('--full', dest='full', action='store_true',
                        help='Full training with metrics, checkpoints, and image samples.')
    parser.add_argument('--train_data', type=str, default='./dataset/nyu_data.zip', help='train dataset path.')
    parser.add_argument('--test_data', type=str, default='./dataset/nyu_test.zip', help='test dataset path.')
    parser.add_argument('--train_tfrecords', type=str, default='./dataset/nyu_data.tfrecords',
                        help='train dataset tfrecords path.')
    parser.add_argument('--result', type=str, default='./result',
                        help='The result directory where the model checkpoints '
                             'will be written.')
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,
                        help='Must choose one from (1) allow_fp32_to_fp16 (2) force_fp16 (3) allow_mix_precision')
    parser.add_argument('--op_select_implmode', default="high_precision", type=str,
                        help='Must choose one from (1) high_precision (2) high_performance')
    parser.add_argument('--is_distributed', default=False, type=str2bool, help='Whether to use multi-npu')
    parser.add_argument('--is_loss_scale', default=True, type=str2bool, help='Whether to use loss scale')
    parser.add_argument('--hcom_parallel', default=False, type=str2bool, help='Whether to use parallel allreduce')
    args = parser.parse_args()

    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["op_select_implmode"].s = tf.compat.as_bytes(args.op_select_implmode)
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
    custom_op.parameter_map["hcom_parallel"].b = args.hcom_parallel

    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    npu_keras_sess = set_keras_session_npu_config(config=sess_config)
    # (npu_sess, npu_shutdown) = init_resource(config=sess_config)

    # Create the model
    print("Create the model")
    model = create_model()

    # Data loaders

    # Use train tfrecords
    train_dataset = create_dataset(args.train_tfrecords, args.bs, is_distributed=args.is_distributed)

    iters_per_epoch = args.steps

    # Training session details
    runID = str(int(time.time())) + '-n' + str(iters_per_epoch) + '-e' + str(args.epochs) + '-bs' + str(
        args.bs) + '-lr' + str(args.lr) + '-device' + str(os.getenv('ASCEND_DEVICE_ID')) + '-' + args.name
    outputPath = os.path.join(args.result, 'models/')
    runPath = outputPath + runID
    pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
    print('Output: ' + runPath)

    learning_rate = tf.Variable(initial_value=tf.constant(args.lr, dtype=tf.float32, shape=[]), trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-7)
    print(args.is_distributed, args.is_loss_scale)

    # if args.is_distributed:
    #     optimizer = npu_distributed_optimizer_wrapper(optimizer)
    # if args.is_loss_scale:
    #     loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=500,
    #                                                            decr_every_n_nan_or_inf=2, incr_ratio=2.0,
    #                                                            decr_ratio=0.8)
    #
    #     optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager, is_distributed=args.is_distributed)

    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=500,
                                                           decr_every_n_nan_or_inf=2, incr_ratio=2.0,
                                                           decr_ratio=0.8)

    optimizer = NPUOptimizer(optimizer, loss_scale_manager, is_distributed=args.is_distributed, is_loss_scale=args.is_loss_scale, is_tailing_optimization=args.is_distributed)

    model.compile(loss=depth_loss_function, optimizer=optimizer)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    if len(args.ckptdir) != 0:
        print("Load ckpt:{}".format(args.ckptdir))
        latest_checkpoint = tf.train.latest_checkpoint(args.ckptdir)
        checkpoint.restore(latest_checkpoint).assert_existing_objects_matched().expect_partial().run_restore_ops()

    print('Ready for training!')
    print('Batch size: {} iters/epoch: {}'.format(args.bs, iters_per_epoch))

    # Callbacks

    callbacks = get_nyu_callbacks(
        test_set=load_test_data(test_data_zip_file=args.test_data) if args.full else None,
        runPath=runPath, checkpoint=checkpoint, lr=args.lr, lr_tf=learning_rate, is_distributed=args.is_distributed)

    start = time.time()

    model.fit(train_dataset, callbacks=callbacks, epochs=args.epochs, steps_per_epoch=iters_per_epoch,
              shuffle=True, verbose=2)

    stop = time.time()

    # Because the model takes too long to compile in the first Sesson run , the performance data may be incorrect

    print('Performance %10.3f sec/iter %10.3f images/sec' % ((stop - start) / (iters_per_epoch * args.epochs),
                                                             (iters_per_epoch * args.epochs * args.bs) / (
                                                                     stop - start)))
    print("Train success")

    evaluate_ckptdir = runPath + '/ckpt_npu'
    tf.io.write_graph(npu_keras_sess.graph, evaluate_ckptdir, 'graph.pbtxt', as_text=True)
    checkpoint.save(file_prefix=evaluate_ckptdir + "/epoch_end/model.ckpt")

    print("Save success!Model save path:{}".format(evaluate_ckptdir))

    # shutdown_resource(npu_sess, npu_shutdown)
    # close_session(npu_sess)
    close_session(npu_keras_sess)


if __name__ == '__main__':
    main()
