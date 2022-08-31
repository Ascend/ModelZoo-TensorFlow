# coding=utf-8
"""Main function for the project."""

from __future__ import absolute_import
from __future__ import division
from npu_bridge.npu_init import *

import os

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

# from ieg import options
# from ieg import utils
# from ieg.dataset_utils import datasets
# from ieg.models.basemodel import BaseModel
# from ieg.models.fsr import FSR
# from ieg.models.l2rmodel import L2R
# from ieg.models.model import IEG

import options
import utils
from dataset_utils import datasets
from models.basemodel import BaseModel
from models.fsr import FSR
from models.l2rmodel import L2R
from models.model import IEG
# import moxing as mox

logger = tf.get_logger()
logger.propagate = False

FLAGS = flags.FLAGS

options.define_basic_flags()
FLAGS.dataset="cifar10_uniform_0.2"
FLAGS.network_name="wrn28-10"
FLAGS.checkpoint_path="/home/TestUser06/IEG_YYW/code/output"
FLAGS.probe_dataset_hold_ratio=0.002
FLAGS.max_epoch=200
#custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("/cache/fusion/fusion_switch.cfg")


def train(model, sess):
  """Training launch function."""
  with sess.as_default():
    model.train()


def evaluation(model, sess):
  """Evaluation launch function."""
  with sess.as_default():
    model.evaluation()


def main(_):

  tf.disable_v2_behavior()
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    None
  strategy = utils.get_distribution_strategy(
      FLAGS.distribution_strategy, tpu_address=FLAGS.tpu)

  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  config.gpu_options.allow_growth = True
  custom_op=config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name="NpuOptimizer"
  custom_op.parameter_map["use_off_line"].b = True
  custom_op.parameter_map['precision_mode'].s=tf.compat.as_bytes("allow_mix_precision")
  config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  config.graph_options.rewrite_options.memory_optimization=RewriterConfig.OFF

  sess = tf.Session(config=config)

  # Creates dataset
  if 'cifar' in FLAGS.dataset:
    dataset = datasets.CIFAR(include_metadata=FLAGS.ds_include_metadata)
  elif 'webvisionmini' in FLAGS.dataset:
    # webvision mini version
    dataset = datasets.WebVision(
        root=os.path.join(FLAGS.dataset_dir, 'tensorflow_datasets'),
        version='webvisionmini-google',
        use_imagenet_as_eval=FLAGS.use_imagenet_as_eval,
        add_strong_aug=FLAGS.method == 'ieg')

  if FLAGS.xm_exp_id:
    if FLAGS.exp_path_pattern:
      pattern = ['id' + str(FLAGS.xm_exp_id)]
      for field in FLAGS.exp_path_pattern.split('+'):
        value = getattr(FLAGS, field)
        pattern.append(field + '=' + str(value))
      FLAGS.checkpoint_path = os.path.join(FLAGS.checkpoint_path,
                                           '+'.join(pattern))
    else:
      FLAGS.checkpoint_path = os.path.join(FLAGS.checkpoint_path,
                                           str(FLAGS.xm_exp_id))
  else:
    FLAGS.checkpoint_path = os.path.join(FLAGS.checkpoint_path, FLAGS.dataset,
                                         FLAGS.network_name)

  if FLAGS.method == 'supervised':
    model = BaseModel(sess=sess, strategy=strategy, dataset=dataset)
  elif FLAGS.method == 'l2r':
    model = L2R(sess=sess, strategy=strategy, dataset=dataset)
  elif FLAGS.method == 'ieg':
    model = IEG(sess=sess, strategy=strategy, dataset=dataset)
  elif FLAGS.method == 'fsr':
    # Hard-code parameters not used for FSR.
    FLAGS.ds_include_metadata = True
    FLAGS.probe_dataset_hold_ratio = 0
    FLAGS.meta_partial_level = 0
    model = FSR(sess=sess, strategy=strategy, dataset=dataset)
  else:
    raise NotImplementedError('{} is not existed'.format(FLAGS.method))

  utils.print_flags(FLAGS)
  if FLAGS.mode == 'evaluation':
    evaluation(model, sess)
  else:
    train(model, sess)

if __name__ == '__main__':
  app.run(main)

