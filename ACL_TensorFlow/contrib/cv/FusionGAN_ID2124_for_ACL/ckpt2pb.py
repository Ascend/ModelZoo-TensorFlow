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
import tensorflow as tf
from model import CGAN
from tensorflow_core.contrib.slim import nets
from tensorflow_core.python.tools import freeze_graph
from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
  weights_spectral_norm,
  l2_norm
)

# atc --model=./good_frozen8.pb --framework=3 --input_shape="images_ir:1,282,372,1;images_vi:1,282,372,1" --output=./toom8 --soc_version=Ascend310
# ./msame --model "./toom8.om" --input "./Nato_camp/vi,./Nato_camp/ir" --output "./out/" --outfmt TXT --debug true

def lrelu(x, leak=0.2):
    """
    x : 输入图像
    """
    return tf.maximum(x, leak * x)


def weights_spectral_norm(weights, u=None, iteration=1, update_collection=None, reuse=False, name='weights_SN'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        if u is None:
            u = tf.get_variable('u', shape=[1, w_shape[-1]], 
            initializer=tf.truncated_normal_initializer(), trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite + 1
        
        u_hat, v_hat, _ = power_iteration(u, iteration)
        
        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
        
        w_mat = w_mat / sigma
        
        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            if not(update_collection == 'NO_OPS'):
                print(update_collection)
                tf.add_to_collection(update_collection, u.assign(u_hat))
            
            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm


def fusion_model(img):
    """
    img : 输入图像
    """
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1", [5, 5, 2, 256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b1", [256], initializer=tf.constant_initializer(0.0))
            conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1, 1, 1, 1], 
            padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ir = lrelu(conv1_ir)
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2", [5, 5, 256, 128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b2", [128], initializer=tf.constant_initializer(0.0))
            conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_ir, weights, strides=[1, 1, 1, 1], 
                    padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ir = lrelu(conv2_ir)
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3", [3, 3, 128, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b3", [64], initializer=tf.constant_initializer(0.0))
            conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ir = lrelu(conv3_ir)
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4", [3, 3, 64, 32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b4", [32], initializer=tf.constant_initializer(0.0))
            conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5", [1, 1, 32, 1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b5", [1], initializer=tf.constant_initializer(0.0))
            conv5_ir= tf.nn.conv2d(conv4_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv5_ir=tf.nn.tanh(conv5_ir)
    return conv5_ir
print("111")
#def main():
CKPT_PATH = './checkpoint/CGAN_120/CGAN.model-8'
output_path = './'
print("222")
tf.compat.v1.reset_default_graph()
print("3333")
# 定义网络的输入节点，输入大小与模型在线测试时一致
images_ir = tf.placeholder(tf.float32, [1, 282, 372, 1], name='images_ir')
images_vi = tf.placeholder(tf.float32, [1, 282, 372, 1], name='images_vi')
input_image=tf.concat([images_ir, images_vi], axis=-1)
# 调用网络模型生成推理图，用法参考slim
print("111")
fusion_image=fusion_model(input_image)
fusion_image_out = tf.identity(fusion_image, name='output')
print('generate:', fusion_image.shape)


with tf.compat.v1.Session() as sess:
    #保存图，在 DST_FOLDER 文件夹中生成tmp_model.pb文件
    # tmp_model.pb文件将作为input_graph给到接下来的freeze_graph函数
    tf.io.write_graph(sess.graph_def, output_path, './pb/tmp_model8.pb')    # 通过write_graph生成模型文件
    freeze_graph.freeze_graph(
            input_graph=os.path.join(output_path, './pb/tmp_model8.pb'),   # 传入write_graph生成的模型文件
            input_saver='',
            input_binary=False,
            input_checkpoint=CKPT_PATH,  # 传入训练生成的checkpoint文件
            output_node_names='output',  # 与重新定义的推理网络输出节点保持一致
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=os.path.join(output_path, 'fusionGAN8.pb'),   # 改为需要生成的推理网络的名称
            clear_devices=False,
            initializer_nodes='')