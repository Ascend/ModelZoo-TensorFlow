# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Training loop for object discovery with Slot Attention."""
from npu_bridge.npu_init import *
import datetime
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
#tf.disable_eager_execution()
import data as data_utils
import model as model_utils
import utils as utils

from tensorflow.python.framework.graph_util import convert_variables_to_constants


FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "/root/checkp/",
                    "Where to save the checkpoints.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 64, "Batch size for the model.")
flags.DEFINE_integer("num_slots", 7, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
flags.DEFINE_integer("num_train_steps", 500000, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 10000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_integer("decay_steps", 100000,
                     "Number of steps for the learning rate decay.")


# We use `tf.function` compilation to speed up execution. For debugging,
# consider commenting out the `@tf.function` decorator.
def train_step(batch, model, optimizer):
  """Perform a single training step."""

  # Get the prediction of the models and compute the loss.
  with tf.GradientTape() as tape:
    preds = model(batch["image"], training=True)
    recon_combined, recons, masks, slots = preds
    loss_value = utils.l2_loss(batch["image"], recon_combined)
    print("-----------recon的类型----------", type(recon_combined))
    del recons, masks, slots  # Unused.

  # Get and apply gradients.
  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))

  return loss_value


def main(argv):
  del argv
  # Hyperparameters of the model.
  batch_size = FLAGS.batch_size
  num_slots = FLAGS.num_slots
  num_iterations = FLAGS.num_iterations
  base_learning_rate = FLAGS.learning_rate
  num_train_steps = FLAGS.num_train_steps
  warmup_steps = FLAGS.warmup_steps
  decay_rate = FLAGS.decay_rate
  decay_steps = FLAGS.decay_steps
  tf.random.set_random_seed(FLAGS.seed)
  #tf.random.set_seed(FLAGS.seed)
  resolution = (128, 128)
  tf.disable_eager_execution()
  sess = tf.Session(config=npu_config_proto())

  # Build dataset iterators, optimizers and model.
  data_iterator = data_utils.build_clevr_iterator(
      batch_size, split="train", resolution=resolution, shuffle=True,
      max_n_objects=6, get_properties=False, apply_crop=True)

  optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)

  model = model_utils.build_model(resolution, batch_size, num_slots,
                                  num_iterations, model_type="object_discovery")

  # Prepare checkpoint manager.
  global_step = tf.Variable(
      0, trainable=False, name="global_step", dtype=tf.int64)
  ckpt = tf.train.Checkpoint(
      network=model, optimizer=optimizer, global_step=global_step)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=ckpt, directory=FLAGS.model_dir, max_to_keep=5)
  # saver = tf.train.Saver(max_to_keep=5)
  # save_path=tf.train.latest_checkpoint(FLAGS.model_dir)

  ckpt.restore(ckpt_manager.latest_checkpoint)
  if ckpt_manager.latest_checkpoint:
    logging.info("Restored from %s", ckpt_manager.latest_checkpoint)
  else:
    logging.info("Initializing from scratch.")

  # if save_path:
  #   saver.restore(sess, save_path=save_path)
  #   logging.info("Restored from %s", save_path)
  # else:
  #   logging.info("Initializing from scratch.")

  start = time.time()
  init=tf.global_variables_initializer()
  sess.run(init)
  for _ in range(num_train_steps):

    val = data_iterator.get_next()
    batch = sess.run(val)
    print("查看batch的类型",type(batch))

    # Learning rate warm-up.
    if global_step < warmup_steps:
      learning_rate = base_learning_rate * tf.cast(
          global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
    else:
      learning_rate = base_learning_rate
    learning_rate = learning_rate * (decay_rate ** (
        tf.cast(global_step, tf.float32) / tf.cast(decay_steps, tf.float32)))
    optimizer.lr = learning_rate.numpy()

    # loss_value = train_step(batch, model, optimizer)
    train_op=train_step(batch, model, optimizer)
    loss=sess.run(train_op)
    print("--------loss的类型--------", type(loss))
    # Update the global step. We update it before logging the loss and saving
    # the model so that the last checkpoint is saved at the last iteration.
    global_step.assign_add(1)

    # Log the training loss.
    if not global_step % 100:
      logging.info("Step: %s, Loss: %.6f, Time: %s",
                   global_step.numpy(), loss_value,
                   datetime.timedelta(seconds=time.time() - start))

    # We save the checkpoints every 1000 iterations.
    if not global_step  % 1000:
      # Save the checkpoint of the model.
      saved_ckpt = ckpt_manager.save()
      #saved_ckpt=saver.save(sess, path, global_step=global_step)
      logging.info("Saved checkpoint: %s", saved_ckpt)
      graphpath = '/root/graphcheck'
      tf.io.write_graph(sess.graph, graphpath, 'graph.pbtxt', as_text=True)
      logging.info('Save graph {} at iteration {}'.format(graphpath, global_step.numpy()))



if __name__ == "__main__":
  npu_keras_sess = set_keras_session_npu_config()
  app.run(main)
  close_session(npu_keras_sess)

