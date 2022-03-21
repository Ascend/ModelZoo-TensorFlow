# MIT License

# Copyright (c) 2018 Deniz Engin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
import tensorflow as tf
from model import CycleGAN
from reader import Reader
from datetime import datetime
import os
import logging
import subprocess
from utils import ImagePool

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size1', 256, 'image size, default: 256')
tf.flags.DEFINE_integer('image_size2', 256, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_float('lambda1', 10.0,
                      'weight for forward cycle loss (X->Y->X), default: 10.0')#可能需要改成DEFINE_float
tf.flags.DEFINE_float('lambda2', 10.0,
                      'weight for backward cycle loss (Y->X->Y), default: 10.0')#可能需要改成DEFINE_float
tf.flags.DEFINE_float('learning_rate', 1e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')

tf.flags.DEFINE_string('X', 'data/tfrecords/hazyImage.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/hazyImage.tfrecords')
tf.flags.DEFINE_string('Y', 'data/tfrecords/clearImage.tfrecords',
                       'Y tfrecords file for training, default: data/tfrecords/clearImage.tfrecords')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('log_file', "logs/train.log",
                       'training log file, default: logs/train.log')


def train():
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    #checkpoints_dir = "checkpoints/{}".format(current_time)
    checkpoints_dir="checkpoints/Hazy2GT"
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass

  graph = tf.Graph()
  with graph.as_default():
    cycle_gan = CycleGAN(
        X_train_file=FLAGS.X,
        Y_train_file=FLAGS.Y,
        batch_size=FLAGS.batch_size,
        image_size1=FLAGS.image_size1,
        image_size2=FLAGS.image_size2,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        lambda1=FLAGS.lambda1,
        lambda2=FLAGS.lambda2,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf
    )
    G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x  = cycle_gan.model()
    optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver()

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  with tf.Session(config=config, graph=graph) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = "checkpoints/Hazy2GT/model.ckpt-200000.meta"
      print(tf.train.latest_checkpoint(checkpoints_dir))
      restore = tf.train.import_meta_graph(meta_graph_path)
      #restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      restore.restore(sess, "checkpoints/Hazy2GT/model.ckpt-200000")
      step = int(meta_graph_path.split("-")[1].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      fake_Y_pool = ImagePool(FLAGS.pool_size)
      fake_X_pool = ImagePool(FLAGS.pool_size)

      while step < 70005:
        # get previously generated images
        fake_y_val, fake_x_val = sess.run([fake_y, fake_x])

        # train
        _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
              sess.run(
                  [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op],
                  feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                             cycle_gan.fake_x: fake_X_pool.query(fake_x_val)}
              )
        )

        train_writer.add_summary(summary, step)
        train_writer.flush()

        if step % 100 == 0:
          logging.info('-----------Step %d:-------------' % step)
          logging.info('  G_loss   : {}'.format(G_loss_val))
          logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
          logging.info('  F_loss   : {}'.format(F_loss_val))
          logging.info('  D_X_loss : {}'.format(D_X_loss_val))
          print('-----------Step %d:-------------' % step)
          print('  G_loss   : {}'.format(G_loss_val))
          print('  D_Y_loss : {}'.format(D_Y_loss_val))
          print('  F_loss   : {}'.format(F_loss_val))
          print('  D_X_loss : {}'.format(D_X_loss_val))

        if step % 10000 == 0:
          save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
          logging.info("Model saved in file: %s" % save_path)
          ##subprocess.call("./create_model.sh gpu")
          bash_command = 'bash create_model.sh gpu'
          os.system(bash_command)
        step += 1

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      logging.info('other exception!')
      coord.request_stop(e)
    finally:
      logging.info('model done!')
      print('training done!')
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      logging.info("Model saved in file: %s" % save_path)
      bash_command = 'bash create_model.sh gpu'
      os.system(bash_command)
      print('model built done!')
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, filename=FLAGS.log_file)
  tf.app.run()

