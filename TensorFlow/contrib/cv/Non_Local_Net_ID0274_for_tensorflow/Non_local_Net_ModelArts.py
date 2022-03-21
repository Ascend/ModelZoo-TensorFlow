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
# ============================================================================
import npu_bridge.npu_init.npu_config_proto
# from npu_bridge.npu_init import *
import argparse
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import moxing as mox
import os
from Non_Local_Net_ID0274_for_TensorFlow.Network import Build_ResNet
from tensorflow.python.tools import freeze_graph
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer

def parse_args():
    desc = 'MAIN'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_url', type=str, default='./dataset', help='dataset_name')
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size')
    return parser.parse_args()

args = parse_args()

data_dir = "/cache/dataset"
os.makedirs(data_dir)
# Copy OBS data to ModelArts container
mox.file.copy_parallel(args.data_url, data_dir)

model_dir = "/cache/result"
os.makedirs(model_dir)
mox.file.copy_parallel(model_dir, args.train_url)


data_path = args.data_url
mnist = input_data.read_data_sets(args.data_url, one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])

X_img = tf.pad(X_img, [[0, 0], [98, 98], [98, 98], [0, 0]])

learning_rate = 0.0001
batch_size = args.batch_size
num_epoches = args.epoch

logit = Build_ResNet(X_img, resnet_size=50)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y))
cost_summ = tf.summary.scalar("cost", cost)

is_correction = tf.equal(tf.argmax(logit, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correction, tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

#freeze_graph
predict_class = tf.argmax(logit, axis=1, output_type=tf.int32, name="output")
#
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
###############################Loss scale ####################################
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = NPUDistributedOptimizer(optimizer)
    # loss_scale_manager = FixedLossScaleManager(loss_scale=1024)
    # loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000,
    #                                                        decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    #
    # optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager, is_distributed=False).minimize(cost)




############################## npu modify #########################
init = tf.global_variables_initializer()

config_proto = tf.ConfigProto()
custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
config = npu_config_proto(config_proto=config_proto)



# with tf.Session() as sess:
with tf.Session(config=config) as sess:
    sess.run(init)
############################## npu modify #########################
    merged_summary = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter("./log")
    writer.add_graph(sess.graph)

    print("Learning start...")
    avg_accs = []
    avg_costs = []
    for epoch in range(num_epoches):
        avg_acc = 0
        avg_cost = 0

        num_batches = int(mnist.train.num_examples / batch_size)
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_x, Y: batch_y}
        start_time = time.time()

        for i in range(num_batches):
            summary, _, c, a = sess.run([merged_summary, optimizer, cost, accuracy], feed_dict=feed_dict)
            writer.add_summary(summary, global_step=i)
            avg_acc += a / num_batches
            avg_cost += c / num_batches

        end_time = time.time()
        steps_per_s = end_time - start_time
        avg_accs.append(avg_acc)
        avg_costs.append(avg_cost)
        print("Epoch: {}\tLoss:{:.9f}\tAccuarcy: {:.4}\tglobal_step/sec:{}".format(epoch+1, avg_cost, avg_acc, steps_per_s))
     ######### tensorflow  ckpt PB#######################
    saver = tf.train.Saver()
    saver.save(sess, args.train_url + "Model/model.ckpt")
    tf.train.write_graph(sess.graph_def, args.train_url, 'PB_Model/graph.pb')
    freeze_graph.freeze_graph(
        input_graph=args.train_url + 'PB_Model/graph.pb',
        input_saver='',
        input_binary=False,
        input_checkpoint=args.train_url + "Model/model.ckpt",
        output_node_names= "output",
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph= args.train_url + 'PB_Model/frozen_model.pb',
        clear_devices=False,
        initializer_nodes=''
    )

print("Learning finished!")
for i in range(len(avg_accs)):
    print("Epoch: {}\tLoss:{:.9f}\tAccuarcy: {:.4}".
          format(i + 1, avg_costs[i], avg_acc[i]))



