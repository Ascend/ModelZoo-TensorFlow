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


# This file based on : https://jmetzen.github.io/notebooks/vae.ipynb
# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, R0902

import os
import numpy as np
import tensorflow as tf
import utils
import model_def
import data_input
import time


#npu
from npu_bridge.npu_init import *
flags = tf.flags
FLAGS = flags.FLAGS
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
#关闭日志级别
#os.system('export ASCEND_SLOG_PRINT_TO_STDOUT=0')
#os.system('export ASCEND_GLOBAL_LOG_LEVEL=3')

#import argparse
#import moxing as mox
# 解析输入参数data_url
#parser = argparse.ArgumentParser()
#parser.add_argument("--data_url", type=str, default="obs://mnist-benny/mnist/")
# config = parser.parse_args()
#config, unparsed = parser.parse_known_args()
# 在ModelArts容器创建数据存放目录
data_dir = "./dataset"
os.makedirs(data_dir)
# OBS数据拷贝到ModelArts容器内
#mox.file.copy_parallel(config.data_url, data_dir)


def main(hparams):
    # Set up some stuff according to hparams
    utils.set_up_dir(hparams.ckpt_dir)
    utils.set_up_dir(hparams.sample_dir)
    utils.print_hparams(hparams)

    # 读取测试集
    from tensorflow.examples.tutorials.mnist import input_data
    # mnist_data = '/home/dataset/mnist'
    mnist_data = FLAGS.data_dir
    mnist = input_data.read_data_sets(mnist_data, one_hot=True)  # ./data/mnist
    test_images = mnist.test.images
    print("test_data", test_images.shape)

    # encode
    x_ph = tf.placeholder(tf.float32, [None, hparams.n_input], name='x_ph')
    z_mean, z_log_sigma_sq = model_def.encoder(hparams, x_ph, 'enc', reuse=False)

    # sample
    eps = tf.random_normal((hparams.batch_size, hparams.n_z), 0, 1, dtype=tf.float32)
    z_sigma = tf.sqrt(tf.exp(z_log_sigma_sq))
    z = z_mean + z_sigma * eps

    # reconstruct
    logits, x_reconstr_mean = model_def.generator(hparams, z, 'gen', reuse=False)

    # generator sampler
    z_ph = tf.placeholder(tf.float32, [None, hparams.n_z], name='x_ph')
    _, x_sample = model_def.generator(hparams, z_ph, 'gen', reuse=True)

    # define loss and update op
    total_loss = model_def.get_loss(x_ph, logits, z_mean, z_log_sigma_sq)
    opt = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
    update_op = opt.minimize(total_loss)

    # Sanity checks
    for var in tf.global_variables():
        print(var.op.name)
    print ('')

    #npu迁移代码
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    # Get a new session
    sess = tf.Session(config=config)

    # Model checkpointing setup
    model_saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Attempt to restore variables from checkpoint
    start_epoch = utils.try_restore(hparams, sess, model_saver)

    # Get data iterator
    iterator = data_input.mnist_data_iteratior()
    print("start training")

    # Training
    min_re = 10000
    best_epoch = 0
    for epoch in range(start_epoch+1, hparams.training_epochs):
        avg_loss = 0.0
        num_batches = hparams.num_samples // hparams.batch_size
        batch_num = 0

        for (x_batch_val, _) in iterator(hparams, num_batches):
            batch_num += 1
            feed_dict = {x_ph: x_batch_val}

            t1 = time.time()
            _, loss_val = sess.run([update_op, total_loss], feed_dict=feed_dict)
            avg_loss += loss_val / hparams.num_samples * hparams.batch_size
            t2 = time.time()
            print('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % (
                (epoch, loss_val, hparams.batch_size / (t2 - t1), (t2 - t1))))

#npu中不需要保存图片
            if batch_num % 100 == 0:
                x_reconstr_mean_val = sess.run(x_reconstr_mean, feed_dict={x_ph: x_batch_val})

                z_val = np.random.randn(hparams.batch_size, hparams.n_z)
                x_sample_val = sess.run(x_sample, feed_dict={z_ph: z_val})

                # utils.save_images(np.reshape(x_reconstr_mean_val, [-1, 28, 28]),
                #                   [10, 10],
                #                   '{}/reconstr_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
                # utils.save_images(np.reshape(x_batch_val, [-1, 28, 28]),
                #                   [10, 10],
                #                   '{}/orig_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
                #
                # utils.save_images(np.reshape(x_sample_val, [-1, 28, 28]),
                #                   [10, 10],
                #                   '{}/sampled_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))


        if epoch % hparams.summary_epoch == 0:
            #测试re
            reconstruct_error = 0
            test_index = np.random.randint(0,1000,size=(100,))
            test_data = test_images[test_index]
            x_reconstr_mean_test = sess.run(x_reconstr_mean, feed_dict={x_ph: test_data})
            # print(type(x_reconstr_mean_test))
            # print(type(test_data))
            # x_reconstr_mean_test = x_reconstr_mean_test[0]
            # test_data = test_data[0]
            reconstruct_error += np.sum(np.square(np.linalg.norm(x_reconstr_mean_test-test_data)))
            reconstruct_error= reconstruct_error/ (784 * 100)
            if(reconstruct_error<min_re):
                min_re = reconstruct_error
                best_epoch = epoch
            reconstruct_error_val = np.sum(np.square(np.linalg.norm(x_reconstr_mean_val - x_batch_val)))/(784*100)

            print(epoch)
            print("================val_reconstruct_error====================", reconstruct_error_val)
            print("================test_reconstruct_error====================",reconstruct_error)
            # print("Epoch:", '%04d' % (epoch), 'Avg loss = {:.9f}'.format(avg_loss))


        if epoch % hparams.ckpt_epoch == 0:
            save_path = os.path.join(hparams.ckpt_dir, 'mnist_vae_model')
            model_saver.save(sess, save_path, global_step=epoch)

    print("===============min_reconstruct_error_in_epoch:%d,reis%.5f===================="%(best_epoch,min_re))

    save_path = os.path.join(hparams.ckpt_dir, 'mnist_vae_model')
    model_saver.save(sess, save_path, global_step=hparams.training_epochs-1)
    #npu迁移
    sess.close()
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--train_url", type=str, default="./output")
    #config = parser.parse_args()
    # 在ModelArts容器创建训练输出目录
    model_dir = "./result"
    os.makedirs(model_dir)
    # 训练结束后，将ModelArts容器内的训练输出拷贝到OBS
    #mox.file.copy_parallel(model_dir, config.train_url)


if __name__ == '__main__':

    HPARAMS = model_def.Hparams()

    HPARAMS.num_samples = 60000
    HPARAMS.learning_rate = 0.001
    HPARAMS.batch_size = 100
    HPARAMS.training_epochs = FLAGS.epochs
    HPARAMS.summary_epoch = 1
    HPARAMS.ckpt_epoch = 5

    HPARAMS.ckpt_dir = './models/mnist-vae/'
    HPARAMS.sample_dir = './samples/mnist-vae/'

    main(HPARAMS)


