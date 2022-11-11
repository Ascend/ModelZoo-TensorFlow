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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MAIN')
    parser.add_argument('--total_epochs',type=int,default=160,help='total_epochs')
    parser.add_argument('--test_iteration',type=int,default=10,help='test_iteration')
    parser.add_argument('--iteration',type=int,default=391,help='iteration')
    parser.add_argument('--data_path',type=str,help='data_path')
    args = parser.parse_args()
    return args

args = parse_args()

weight_decay = 0.0001
momentum = 0.9
init_learning_rate = 0.01
batch_size = 128
iteration = args.iteration
# 128 * 391 ~ 50,000
total_epochs = args.total_epochs
test_iteration = args.test_iteration
data_dir = args.data_path



if __name__ == '__main__':
    def Evaluate(sess):
        test_acc = 0.0
        test_loss = 0.0
        test_pre_index = 0
        add = 1000
    
        for it in range(test_iteration):
            test_batch_x = test_x[test_pre_index: test_pre_index + add]
            test_batch_y = test_y[test_pre_index: test_pre_index + add]
            test_pre_index = test_pre_index + add
    
            test_feed_dict = {
                x: test_batch_x,
                label: test_batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: False
            }
    
            loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)
    
            test_loss += loss_
            test_acc += acc_
    
        test_loss /= test_iteration  # average loss
        test_acc /= test_iteration  # average accuracy
    
        summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                    tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
    
        return test_acc, test_loss, summary
    import os
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
    from seresnetv2 import seresnet_v2
    from cifar10 import *
    import tensorflow as tf
    from npu_bridge.npu_init import *
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    #custom_op.parameter_map["dynamic_input"].b = True
    #custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    custom_op.parameter_map["use_off_line"].b = True  # 必须显式开启，在昇腾AI处理器执行训练
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    
    #溢出检测
    #custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/HwHiAiUser/output")
    #custom_op.parameter_map["enable_dump_debug"].b = True
    #custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
    
    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = color_preprocessing(train_x, test_x)
    
    # image_size = 32, img_channels = 3, class_num = 10 in cifar10
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
    label = tf.compat.v1.placeholder(tf.float32, shape=[None, class_num])
    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    #模型
    logits = seresnet_v2(x, 110)
    
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    
    
    #loss—scale
    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    opt_tmp = npu_tf_optimizer(
         tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True))
    optimizer = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)
    
    #optimizer = npu_tf_optimizer(
    #    tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True))
    
    train = optimizer.minimize(cost + l2_loss * weight_decay)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(tf.global_variables())
    
    
    
    
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
    
        summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    
        epoch_learning_rate = init_learning_rate
        for epoch in range(1, total_epochs + 1):
            if epoch == 2:
                epoch_learning_rate = 0.1
    
            if epoch == 80:
                epoch_learning_rate = 0.01
    
            if epoch == 120:
                epoch_learning_rate = 0.001
    
    
    
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
    
            for step in range(1, iteration + 1):
                start = time.time()
                if pre_index + batch_size < 50000:
                    batch_x = train_x[pre_index: pre_index + batch_size]
                    batch_y = train_y[pre_index: pre_index + batch_size]
                else:
                    batch_x = train_x[pre_index:]
                    batch_y = train_y[pre_index:]
    
                batch_x = data_augmentation(batch_x)
    
                train_feed_dict = {
                    x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }
    
                _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)
                step_time = time.time() - start
                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size
                print(step_time)
            train_loss /= iteration  # average loss
            train_acc /= iteration  # average accuracy
    
            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                              tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
    
            #test_acc, test_loss, test_summary = Evaluate(sess)
    
            summary_writer.add_summary(summary=train_summary, global_step=epoch)
            #summary_writer.add_summary(summary=test_summary, global_step=epoch)
            summary_writer.flush()
    
            line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f" % (
                epoch, total_epochs, train_loss, train_acc)
            print(line)
    
            with open('logs.txt', 'a') as f:
                f.write(line)
    
            saver.save(sess=sess, save_path='model/senet110.ckpt')



