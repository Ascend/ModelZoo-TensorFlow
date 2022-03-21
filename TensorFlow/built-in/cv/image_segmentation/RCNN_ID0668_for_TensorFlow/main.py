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
from read_data import *
import numpy as np
import tensorflow as tf
import os
from RCNN import *
import datetime
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--data_path', dest='data_path', default='./data/', help='path of the dataset')
parser.add_argument('--precision_mode', dest='precision_mode', default='allow_mix_precision', help='precision mode')
parser.add_argument('--over_dump', dest='over_dump', default='False', help='if or not over detection')
parser.add_argument('--over_dump_path', dest='over_dump_path', default='./overdump', help='over dump path')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', default='False', help='data dump flag')
parser.add_argument('--data_dump_step', dest='data_dump_step', default='10', help='data dump step')
parser.add_argument('--data_dump_path', dest='data_dump_path', default='./datadump', help='data dump path')
parser.add_argument('--profiling', dest='profiling', default='False', help='if or not profiling for performance debug')
parser.add_argument('--profiling_dump_path', dest='profiling_dump_path', default='./profiling', help='profiling path')
parser.add_argument('--autotune', dest='autotune', default='False', help='whether to enable autotune, default is False')

parser.add_argument('--max_step', dest='max_step', type=int, default=100000, help='# of step for training')
parser.add_argument('--save_interval', dest='save_interval', type=int, default=10000, help='# of interval to save  model')
parser.add_argument('--modeldir', dest='modeldir', default='./ckpt', help='ckpt dir')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0005, help='learning rate')
parser.add_argument('--loss_scale', dest='loss_scale', default='False', help='enable loss scale ,default is False')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='# images in batch')
parser.add_argument('--log_interval', dest='log_interval', type=int, default=100, help='# of interval to print log')
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    log_dir = './logdir'
    snapshot_interval = args.save_interval
    snapshot_dir = args.modeldir
    max_iter = args.max_step
    log_interval = args.log_interval

    lr = args.learning_rate

    file = ("%s/train_32x32.mat" %(args.data_path))
    X_raw, y_raw = getData(filename=file)
    n_train = X_raw.shape[0]
    y_raw[y_raw == 10] = 0
    y_raw = np.reshape(y_raw, (n_train,))

    config = npu_config_proto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
    if args.data_dump_flag.strip() == "True":
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.data_dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(args.data_dump_step)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
    if args.over_dump.strip() == "True":
        # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.over_dump_path)
        # enable_dump_debug：是否开启溢出检测功能
        custom_op.parameter_map["enable_dump_debug"].b = True
        # dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
        custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
    if args.profiling.strip() == "True":
        custom_op.parameter_map["profiling_mode"].b = False
        profilingvalue = (
                '{"output":"%s","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":""}' % (
            args.profiling_dump_path))
        custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(profilingvalue)

    with tf.Session(config=config) as sess:
        X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.int32, shape=(None,))
        rcnn = RCNN(time=3, K=192, p=0.9, numclass=10, is_training=True)
        loss, summary_op, acc, _ = rcnn.buile_model(X, y)

        if args.loss_scale.strip() == "True":
            opt = npu_tf_optimizer(tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-8)).minimize(loss)
            loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                                   decr_every_n_nan_or_inf=2, decr_ratio=0.5)
            optimizer = NPULossScaleOptimizer(opt, loss_scale_manager)
        else:
            optimizer = npu_tf_optimizer(tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-8)).minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)

        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots

        writer = tf.summary.FileWriter(log_dir, sess.graph)
        np.random.seed(0)
        loss_mean = 0
        acc_mean = 0
        start = datetime.datetime.now()
        t = 0
        for n_iter in range(max_iter):
            index = np.random.choice(n_train, args.batch_size, replace=True)
            image = X_raw[index]
            labels = y_raw[index]
            # print(image.shape)
            start_time = time.time()
            loss_batch, summary_op_batch, acc_batch, _ = sess.run([loss, summary_op, acc, optimizer], feed_dict={X:image, y:labels})
            loss_mean += loss_batch
            acc_mean += acc_batch
            t += (time.time() - start_time)
            if (n_iter + 1) % log_interval == 0 or (n_iter + 1) == max_iter:
                loss_mean = loss_mean/(log_interval*1.0)
                acc_mean = acc_mean/(log_interval*1.0)
                batch_time = datetime.datetime.now()
                print(
                    "time: {},iter = {}\n\tloss = {}, accuracy (cur) = {} ,perf = {}".format(batch_time - start, n_iter + 1, loss_mean,
                                                                                   acc_mean, t))
                loss_mean = 0
                acc_mean = 0
                t = 0

            writer.add_summary(summary_op_batch, global_step=n_iter)

            if (n_iter + 1) % snapshot_interval == 0 or (n_iter + 1) == max_iter:
                snapshot_file = os.path.join(snapshot_dir, "%08d" % (n_iter + 1))
                snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
                print('snapshot saved to ' + snapshot_file)

    end = datetime.datetime.now()
    print("sum time: {}".format(end - start))
    writer.close()

if __name__ == '__main__':
    main()