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

''' --为了调用cycleGAN初始化一个args-- '''
FLAGS = tf.flags.FLAGS
# 输入ckpt：/home/ma-user/modelarts/inputs/data_url_0/model_ckpt/
tf.flags.DEFINE_string('checkpoint_dir', 'model_ckpt', 'checkpoints directory path')
# 输出pb：/home/ma-user/modelarts/outputs/train_url_0/pb
tf.flags.DEFINE_string('pb_dir', '/home/ma-user/modelarts/outputs/train_url_0/pb', 'pb directory')
tf.flags.DEFINE_string('dataset_dir', 'avatar_data', 'path of the dataset')
tf.flags.DEFINE_string("data_dir", "/home/ma-user/modelarts/inputs/data_url_0", '')
tf.flags.DEFINE_string("model_dir", "/home/ma-user/modelarts/outputs/train_url_0/", '')
tf.flags.DEFINE_string('XtoY_model', '/home/ma-user/modelarts/outputs/train_url_0/pb/real2avatar.pb', 'XtoY model name, default: real2avatar.pb')
tf.flags.DEFINE_string('YtoX_model', '/home/ma-user/modelarts/outputs/train_url_0/pb/avatar2real.pb', 'YtoX model name, default: avatar2real.pb')

tf.flags.DEFINE_integer('fine_size', 256, 'image size, default: 256')
tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_integer('ndf', 64, 'number of discri filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance', '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('batch_size', 1, '# images in batch, default: 1')
tf.flags.DEFINE_float('L1_lambda', 10.0, 'weight on L1 term in objective')

tf.flags.DEFINE_integer('input_nc', 1, '# of input image channels')
tf.flags.DEFINE_integer('output_nc', 1, '# of output image channels')
tf.flags.DEFINE_integer("max_size", 50, 'max size of image pool, 0 means do not use image pool')

tf.flags.DEFINE_boolean("use_resnet", True, '')
tf.flags.DEFINE_boolean("use_lsgan", True, '')
tf.flags.DEFINE_string("phase", 'Train', '')

def export_graph(model_name, XtoY=True):
    print(" [*] Reading checkpoint...")
    target_ckpt = "%s_%s" % (FLAGS.dataset_dir, FLAGS.fine_size)
    checkpoint_dir = '{}/{}/{}'.format(FLAGS.data_dir, FLAGS.checkpoint_dir, target_ckpt) # ckpt所在目录
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # 载入模型，不需要提供模型的名字，会通过 checkpoint 文件定位到最新保存的模型
    #print(ckpt)
    # model_checkpoint_path: "/home/ma-user/modelarts/outputs/train_url_0/checkpoint/avatar_data_256/cyclegan.model-85002"
    # all_model_checkpoint_paths: "/home/ma-user/modelarts/outputs/train_url_0/checkpoint/avatar_data_256/cyclegan.model-83002"
    # all_model_checkpoint_paths: "/home/ma-user/modelarts/outputs/train_url_0/checkpoint/avatar_data_256/cyclegan.model-84002"
    # all_model_checkpoint_paths: "/home/ma-user/modelarts/outputs/train_url_0/checkpoint/avatar_data_256/cyclegan.model-85002"
    if ckpt and ckpt.model_checkpoint_path:
        print("success")
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # 得ckpt文件路径
        # print(ckpt_name) # cyclegan.model-85002
        model_path = os.path.join(checkpoint_dir, ckpt_name) #model_ckpt完整路径
    else:
        print("failure")
    tf.reset_default_graph()

    with tf.Session() as sess:
        cycle_gan = Cyclegan(sess, FLAGS)
        # 模型输入
        input_image = tf.placeholder(tf.float32,
                                 shape=[1, FLAGS.fine_size, FLAGS.fine_size, 1],
                                 name='input_image')
        # 模型输出
        output_image = cycle_gan.sample(input_image, XtoY)


        # 设置输出的接口名字，为了之后的调用pb的时候使用
        output_image = tf.identity(output_image, name='output_image')

        # 加载模型
        restore_saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        restore_saver.restore(sess, model_path)

        # 保存图，在./pb_model文件夹中生成model.pb文件，作为input_graph给到接下来的freeze_graph函数
        if not os.path.exists(FLAGS.pb_dir):
            os.makedirs(FLAGS.pb_dir)
        tf.train.write_graph(sess.graph_def, FLAGS.pb_dir, 'model.pb') # 通过write_graph生成模型文件
        # 把图和参数结构一起
        freeze_graph.freeze_graph(
            input_graph=os.path.join(FLAGS.pb_dir, 'model.pb'),
            input_saver='',
            input_binary=False,
            input_checkpoint=model_path,
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
