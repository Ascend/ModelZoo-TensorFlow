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


""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python export_graph.py --checkpoint_dir ./model_ckpt/avatar_data_256 \
                       --XtoY_model real2avatar.pb \
                       --YtoX_model avatar2real.pb \
                       --fine_size 256
"""

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from model import *

# 为了调用cycleGAN初始化一个args
FLAGS = tf.flags.FLAGS
#/home/ma-user/modelarts/inputs/data_url_0/
tf.flags.DEFINE_string('checkpoint_dir', '/home/ma-user/modelarts/inputs/data_url_0/model_ckpt/', 'checkpoints directory path')
# /home/ma-user/modelarts/outputs/train_url_0/sample
tf.flags.DEFINE_string('pb_dir', '/home/ma-user/modelarts/outputs/train_url_0/pb',
                       'pb directory')
tf.flags.DEFINE_string('XtoY_model', '/home/ma-user/modelarts/outputs/train_url_0/pb/real2avatar.pb', 'XtoY model name, default: real2avatar.pb')
tf.flags.DEFINE_string('YtoX_model', '/home/ma-user/modelarts/outputs/train_url_0/pb/avatar2real.pb', 'YtoX model name, default: avatar2real.pb')

tf.flags.DEFINE_integer('fine_size', '256', 'image size, default: 256')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_integer('ndf', 64,
                        'number of discri filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('batch_size', 1, '# images in batch, default: 1')
tf.flags.DEFINE_float('L1_lambda', 10.0, 'weight on L1 term in objective')
tf.flags.DEFINE_float("alpha", 0.33, "loss weight for two datasets")
tf.flags.DEFINE_float("beta1", 0.5, 'momentum term of adam')
tf.flags.DEFINE_integer('input_nc', 1, '# of input image channels')
tf.flags.DEFINE_integer('output_nc', 1, '# of output image channels')

tf.flags.DEFINE_string('dataset_dir', 'avatar_data', 'path of the dataset')
tf.flags.DEFINE_string("data_dir", "/home/ma-user/modelarts/inputs/data_url_0/", '')
tf.flags.DEFINE_string("model_dir", "/home/ma-user/modelarts/outputs/train_url_0/", '')
tf.flags.DEFINE_integer("max_size", 50, 'max size of image pool, 0 means do not use image pool')
tf.flags.DEFINE_boolean("use_resnet", True, '')
tf.flags.DEFINE_boolean("use_lsgan", True, '')
tf.flags.DEFINE_string("phase", 'Train', '')
tf.flags.DEFINE_boolean("use_fp16", False, '')
tf.flags.DEFINE_integer("avatar_loss_scale", 1024, '')

def export_graph(model_name, XtoY=True):
    print(" [*] Reading checkpoint...")
    # step = 1002
    model_dir = "%s_%s" % (FLAGS.dataset_dir, FLAGS.fine_size)
    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir) # 检查目录下ckpt文件状态是否可用
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        print("success")
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # 得ckpt文件路径
        print(ckpt_name)
    else:
        print("failure")
    tf.reset_default_graph()

    with tf.Session() as sess:
        cycle_gan = Cyclegan(sess, FLAGS)
        # set inputs node
        input_image = tf.placeholder(tf.float32,
                                 shape=[FLAGS.fine_size, FLAGS.fine_size, 3],
                                 name='input_image')
        # set output node
        temp = cycle_gan.sample_model("./", 1, 1)
        if XtoY:
            output_image = temp[0]
        else:
            output_image = temp[1]

        output_image = tf.identity(output_image, name='output_image')

        restore_saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        restore_saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

        if not os.path.exists(FLAGS.pb_dir):
            os.makedirs(FLAGS.pb_dir)
        # 保存图，在./pb_model文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.train.write_graph(sess.graph_def, FLAGS.pb_dir, 'model.pb') # 通过write_graph生成模型文件

        freeze_graph.freeze_graph(
            input_graph=os.path.join(FLAGS.pb_dir, 'model.pb'),
            input_saver='',
            input_binary=False,
            input_checkpoint=os.path.join(checkpoint_dir, ckpt_name),
            output_node_names='output_image',  # graph outputs node
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=model_name,  # graph outputs name
            clear_devices=False,
            initializer_nodes='')
        print("done")



def main(_):
    print('Export XtoY model...')
    export_graph(FLAGS.XtoY_model, XtoY=True)
    print('Export YtoX model...')
    export_graph(FLAGS.YtoX_model, XtoY=False)


if __name__ == '__main__':
    tf.app.run()