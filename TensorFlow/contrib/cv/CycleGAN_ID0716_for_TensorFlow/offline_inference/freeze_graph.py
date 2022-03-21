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


""" Freeze variables and convert a generator networks to a GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python3 freeze_graph.py --checkpoint_dir checkpoints/20170424-1152/model.ckpt-100000 \
                       --XtoY_model horse2zebra.pb \
                       --YtoX_model zebra2horse.pb \
                       --image_size 256
"""

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from model import CycleGAN

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', '../20210726-1013/model.ckpt-200000', 'checkpoints directory path')
tf.flags.DEFINE_string('XtoY_model', 'horse2zebra.pb', 'XtoY model name, default: horse2zebra.pb')
tf.flags.DEFINE_string('YtoX_model', 'zebra2horse.pb', 'YtoX model name, default: zebra2horse.pb')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')


def export_graph(model_name, XtoY=True):
    tf.reset_default_graph()

    cycle_gan = CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)
    input_image = tf.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3], name='input_image')
    if XtoY:
        output_image = cycle_gan.G.sample_for_eval(tf.expand_dims(input_image, 0))
    else:
        output_image = cycle_gan.F.sample_for_eval(tf.expand_dims(input_image, 0))

    output_image = tf.identity(output_image, name='output_image')
    restore_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_saver.restore(sess, FLAGS.checkpoint_dir)
        tf.train.write_graph(sess.graph_def, './', 'model.pb')

        freeze_graph.freeze_graph(
            input_graph='./model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=FLAGS.checkpoint_dir,
            output_node_names='output_image',  # graph outputs node
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=model_name,  # graph outputs name
            clear_devices=False,
            initializer_nodes='')
        print("done")


def main(unused_argv):
    print('Export XtoY model...')
    export_graph(FLAGS.XtoY_model, XtoY=True)
    print('Export YtoX model...')
    export_graph(FLAGS.YtoX_model, XtoY=False)


if __name__ == '__main__':
    tf.app.run()
