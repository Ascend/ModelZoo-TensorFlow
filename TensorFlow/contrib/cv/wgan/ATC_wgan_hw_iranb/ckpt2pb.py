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

import os
import sys

sys.path.append("..")
print(os.getcwd())
import time
import argparse
import importlib
import tensorflow as tf
import tensorflow.contrib as tc

os.environ["SLOG_PRINT_TO_STDOUT"] = "1"
from tensorflow.python.tools import freeze_graph

# from visualize import *


class WassersteinGAN(object):
    def __init__(self, g_net, d_net, x_sampler, z_sampler, data, model):
        self.model = model
        self.data = data
        self.g_net = g_net
        self.d_net = d_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim
        self.x = tf.placeholder(tf.float32, [1, self.x_dim], name="x")
        self.z = tf.placeholder(tf.float32, [1, self.z_dim], name="z")

        self.x_ = self.g_net(self.z)

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)

        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(2.5e-5),
            weights_list=[
                var for var in tf.global_variables() if "weights" in var.name
            ],
        )
        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
                self.d_loss_reg, var_list=self.d_net.vars
            )
            self.g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
                self.g_loss_reg, var_list=self.g_net.vars
            )

        self.d_clip = [
            v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_net.vars
        ]

        sess_config = tf.compat.v1.ConfigProto()
        self.sess = tf.Session(config=sess_config)

    def train(self, batch_size=64, num_batches=60000 // 64):
        plt.ion()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, num_batches):
            d_iters = 5
            if t % 500 == 0 or t < 25:
                d_iters = 100

            for _ in range(0, d_iters):
                bx = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)
                self.sess.run(self.d_clip)
                self.sess.run(self.d_rmsprop, feed_dict={self.x: bx, self.z: bz})

            bz = self.z_sampler(batch_size, self.z_dim)
            self.sess.run(self.g_rmsprop, feed_dict={self.z: bz, self.x: bx})

            if t % 100 == 0:
                bx = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)

                d_loss = self.sess.run(self.d_loss, feed_dict={self.x: bx, self.z: bz})
                g_loss = self.sess.run(self.g_loss, feed_dict={self.z: bz, self.x: bx})
                print(
                    "Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]"
                    % (t, time.time() - start_time, d_loss - g_loss, g_loss)
                )

            # if t % 100 == 0:
            #     bz = self.z_sampler(batch_size, self.z_dim)
            #     bx = self.sess.run(self.x_, feed_dict={self.z: bz})
            #     bx = xs.data2img(bx)
            #     fig = plt.figure(self.data + '.' + self.model)
            #     grid_show(fig, bx, xs.shape)
            #     fig.savefig('logs/{}/{}.pdf'.format(self.data, t/100))

    def save(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        saver = tf.train.Saver()
        sess = self.sess
        saver.save(sess, save_dir)
        print("save model to: ", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data", type=str, default="lsun", help="mnist or lsun")
    parser.add_argument("--model", type=str, default="dcgan")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--ckpt", type=str, default="./ckpt/")
    parser.add_argument(
        "--iters", type=int, default=300000 // 64 * 60
    )  # 60000 // 64 * 20 for mnist  300000 // 64 * 60 for lsun
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + "." + args.model)
    xs = data.DataSampler()
    zs = data.NoiseSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()
    wgan = WassersteinGAN(g_net, d_net, xs, zs, args.data, args.model)
    # print(wgan.x_) # lsun/dcgan/g_net/Conv2d_transpose_3/Tanh:0
    # exit(0)
    # print("*******************")
    # convert to pb
    ckpt_path = "/root/code/wgan/ATC_wgan/ckpt/save_dir"

    graph = tf.get_default_graph()
    op = graph.get_operations()
    for i, m in enumerate(op):
        try:
            print("index:", i, m.values()[0])
        except:
            break
    # exit(0)
    with tf.Session() as sess:
        gd = sess.graph.as_graph_def()

        # for node in gd.node:
        #     if node.op == 'RefSwitch':
        #         node.op = 'Switch'
        #         for index in xrange(len(node.input)):
        #             if 'moving_' in node.input[index]:
        #                 node.input[index] = node.input[index] + '/read'
        #     elif node.op == 'AssignSub':
        #         node.op = 'Sub'
        #         if 'use_locking' in node.attr: del node.attr['use_locking']

        tf.train.write_graph(gd, "./pbModel", "wgan.pb")
        freeze_graph.freeze_graph(
            input_graph="./pbModel/wgan.pb",
            input_saver="",
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names="lsun/dcgan/g_net/Conv2d_transpose_3/Tanh",
            restore_op_name="save/restore_all",
            filename_tensor_name="save/Const:0",
            output_graph="./pbModel/wgan_tf.pb",
            clear_devices=False,
            initializer_nodes="",
        )
    print("done")
