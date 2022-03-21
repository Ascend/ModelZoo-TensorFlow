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
python export_graph.py --checkpoint_dir checkpoints/20170424-1152 \
                       --XtoY_model apple2orange.pb \
                       --YtoX_model orange2apple.pb \
                       --image_size 256
"""

import tensorflow as tf
from model import CycleGAN


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', '', 'checkpoints directory path')
tf.flags.DEFINE_string('XtoY_model', 'apple2orange.pb', 'XtoY model name, default: apple2orange.pb')
tf.flags.DEFINE_string('YtoX_model', 'orange2apple.pb', 'YtoX model name, default: orange2apple.pb')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')


def export_graph(model_name, XtoY=True):
    graph = tf.Graph()

    with graph.as_default():
        cycle_gan = CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)

        input_image = tf.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3], name='input_image')
        cycle_gan.model()
        if XtoY:
            output_image = cycle_gan.G.sample(tf.expand_dims(input_image, 0))
        else:
            output_image = cycle_gan.F.sample(tf.expand_dims(input_image, 0))

        output_image = tf.identity(output_image, name='output_image')
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        restore_saver.restore(sess, latest_ckpt)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [output_image.op.name])

        tf.train.write_graph(output_graph_def, 'pretrained', model_name, as_text=False)


def main(unused_argv):
    print('Export XtoY model...')
    export_graph(FLAGS.XtoY_model, XtoY=True)
    print('Export YtoX model...')
    export_graph(FLAGS.YtoX_model, XtoY=False)


if __name__ == '__main__':
    tf.app.run()
