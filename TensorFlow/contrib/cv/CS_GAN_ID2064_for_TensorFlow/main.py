# Copyright 2019 DeepMind Technologies Limited and Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
import os

import time
from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import gan
import image_metrics
import utils
import file_utils
from tensorflow.contrib.learn.python.learn import monitored_session as ms
from tensorflow.train import ChiefSessionCreator
flags.DEFINE_integer(
    'num_training_iterations', 200000,
    'Number of training iterations.')
flags.DEFINE_integer(
    'batch_size', 64, 'Training batch size.')
flags.DEFINE_integer(
    'num_latents', 128, 'The number of latents')
flags.DEFINE_integer(
    'summary_every_step', 1000,
    'The interval at which to log debug ops.')
flags.DEFINE_integer(
    'image_metrics_every_step', 2000,
    'The interval at which to log (expensive) image metrics.')
flags.DEFINE_integer(
    'export_every', 10,
    'The interval at which to export samples.')
flags.DEFINE_integer(
    'num_eval_samples', 10000,
    'The number of samples used to evaluate FID/IS')
flags.DEFINE_string(
    'dataset', 'cifar', 'The dataset used for learning (cifar|mnist.')
flags.DEFINE_float(
    'optimisation_cost_weight', 3., 'weight for latent optimisation cost.')
flags.DEFINE_integer(
    'num_z_iters', 3, 'The number of latent optimisation steps.'
    'It falls back to vanilla GAN when num_z_iters is set to 0.')
flags.DEFINE_float(
    'z_step_size', 0.01, 'Step size for latent optimisation.')
flags.DEFINE_string(
    'z_project_method', 'norm', 'The method to project z.')
flags.DEFINE_string(
    'output_dir', 'cs_gan/cs_gan/output', 'Location where to save output files.')
flags.DEFINE_float('disc_lr', 2e-4, 'Discriminator Learning rate.')
flags.DEFINE_float('gen_lr', 2e-4, 'Generator Learning rate.')
flags.DEFINE_bool(
    'run_real_data_metrics', False,
    'Whether or not to run image metrics on real data.')
flags.DEFINE_bool(
    'run_sample_metrics', True,
    'Whether or not to run image metrics on samples.')


FLAGS = flags.FLAGS

# Log info level (for Hooks).
tf.logging.set_verbosity(tf.logging.INFO)


def main(argv):
  del argv

  utils.make_output_dir(FLAGS.output_dir)
  data_processor = utils.DataProcessor()
  images = utils.get_train_dataset(data_processor, FLAGS.dataset,
                                   FLAGS.batch_size)

  logging.info('Generator learning rate: %d', FLAGS.gen_lr)
  logging.info('Discriminator learning rate: %d', FLAGS.disc_lr)

  # Construct optimizers.
  disc_optimizer = tf.train.AdamOptimizer(FLAGS.disc_lr, beta1=0.5, beta2=0.999)
  gen_optimizer = tf.train.AdamOptimizer(FLAGS.gen_lr, beta1=0.5, beta2=0.999)

  # Create the networks and models.
  generator = utils.get_generator(FLAGS.dataset)
  metric_net = utils.get_metric_net(FLAGS.dataset)
  model = gan.GAN(metric_net, generator,
                  FLAGS.num_z_iters, FLAGS.z_step_size,
                  FLAGS.z_project_method, FLAGS.optimisation_cost_weight)
  prior = utils.make_prior(FLAGS.num_latents)
  generator_inputs = prior.sample(FLAGS.batch_size)

  model_output = model.connect(images, generator_inputs)
  optimization_components = model_output.optimization_components
  debug_ops = model_output.debug_ops
  samples = generator(generator_inputs, is_training=False)

  global_step = tf.train.get_or_create_global_step()
  # We pass the global step both to the disc and generator update ops.
  # This means that the global step will not be the same as the number of
  # iterations, but ensures that hooks which rely on global step work correctly.
  def open_loss_scale(bert_loss_scale,opt, key):
      opt_tmp = opt
      if bert_loss_scale == 0:
          # loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
          #                                                        decr_every_n_nan_or_inf=2, decr_ratio=0.5)
          loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 10, incr_every_n_steps=100,
                                                                 decr_every_n_nan_or_inf=2, decr_ratio=0.8)
          print("lossScale type: exponential")
      elif bert_loss_scale >= 1:
          loss_scale_manager = FixedLossScaleManager(loss_scale=bert_loss_scale)
      else:
          raise ValueError("Invalid loss scale: %d" % self.bert_loss_scale)
      #self.mmgr[key] = loss_scale_manager
      # device数是否大于1，如果大于1，进行分布式训练
      # if ops_adapter.size() > 1:
      #     opt_tmp = NPUDistributedOptimizer(opt_tmp)
      #     opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager, is_distributed=True)
      # else:
      opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)
      return opt

  disc_optimizer = open_loss_scale(1, disc_optimizer ,'D')
  gen_optimizer = open_loss_scale(1, gen_optimizer ,'G')


  disc_update_op = disc_optimizer.minimize(
      optimization_components['disc'].loss,
      var_list=optimization_components['disc'].vars,
      global_step=global_step)

  gen_update_op = gen_optimizer.minimize(
      optimization_components['gen'].loss,
      var_list=optimization_components['gen'].vars,
      global_step=global_step)

  # Get data needed to compute FID. We also compute metrics on
  # real data as a sanity check and as a reference point.
  eval_real_data = utils.get_real_data_for_eval(FLAGS.num_eval_samples,
                                                FLAGS.dataset,
                                                split='train')

  def sample_fn(x):
    return utils.optimise_and_sample(x, module=model,
                                     data=None, is_training=False)[0]

  if FLAGS.run_sample_metrics:
    sample_metrics = image_metrics.get_image_metrics_for_samples(
        eval_real_data, sample_fn,
        prior, data_processor,
        num_eval_samples=FLAGS.num_eval_samples)
  else:
    sample_metrics = {}

  if FLAGS.run_real_data_metrics:
    data_metrics = image_metrics.get_image_metrics(
        eval_real_data, eval_real_data)
  else:
    data_metrics = {}

  sample_exporter = file_utils.FileExporter(
      os.path.join(FLAGS.output_dir, 'samples'))

  # Hooks.
  debug_ops['it'] = global_step
  # Abort training on Nans.
  nan_disc_hook = tf.train.NanTensorHook(optimization_components['disc'].loss)
  nan_gen_hook = tf.train.NanTensorHook(optimization_components['gen'].loss)
  # Step counter.
  step_conter_hook = tf.train.StepCounterHook()

  checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      checkpoint_dir=utils.get_ckpt_dir(FLAGS.output_dir), save_secs=10 * 60)

  loss_summary_saver_hook = tf.train.SummarySaverHook(
      save_steps=FLAGS.summary_every_step,
      output_dir=os.path.join(FLAGS.output_dir, 'summaries'),
      summary_op=utils.get_summaries(debug_ops))

  metrics_summary_saver_hook = tf.train.SummarySaverHook(
      save_steps=FLAGS.image_metrics_every_step,
      output_dir=os.path.join(FLAGS.output_dir, 'summaries'),
      summary_op=utils.get_summaries(sample_metrics))

  # hooks = [checkpoint_saver_hook, metrics_summary_saver_hook,
  #          nan_disc_hook, nan_gen_hook, step_conter_hook,
  #          loss_summary_saver_hook]

  hooks = [checkpoint_saver_hook, nan_disc_hook, nan_gen_hook, step_conter_hook]

  config = tf.ConfigProto()
  custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes('allow_mix_precision')
  #custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes('force_fp32')
  config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
  config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
  custom_op.parameter_map["customize_dtypes"].s = tf.compat.as_bytes("switch_config.txt")
  custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/ma-user/modelarts/outputs/train_url_0/")
  # enable_dump_debug：是否开启溢出检测功能
  custom_op.parameter_map["enable_dump_debug"].b = True
  # dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
  custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")

  #init = tf.global_variables_initializer()
  # Start training.   session_creator=ChiefSessionCreator(config=config),
  #init_op = tf.global_variables_initializer()
  with tf.train.MonitoredSession(session_creator=ChiefSessionCreator(config=config),hooks=hooks) as sess:
    logging.info('starting training')
    #tf.reset_default_graph()
    #sess.run(tf.global_variables_initializer())
    #sess.run(init_op)
    #init = tf.global_variables_initializer()
    #sess.run(init)

    for key, value in sess.run(data_metrics).items():
      logging.info('%s: %d', key, value)
      print(key,value)


    for i in range(FLAGS.num_training_iterations):

      start_time = time.time()
      _, dis_loss = sess.run([disc_update_op, optimization_components['disc'].loss])
      _, gen_loss = sess.run([gen_update_op, optimization_components['gen'].loss])
      end_time = time.time()

      print('step: {}, disc_loss: {}, gen_loss: {}, time: {}'.format(i, dis_loss, gen_loss, end_time - start_time))

      if i % FLAGS.export_every == 0:
         samples_np, data_np = sess.run([samples, images])
         # Create an object which gets data and does the processing.
         data_np = data_processor.postprocess(data_np)
         samples_np = data_processor.postprocess(samples_np)
         sample_exporter.save(samples_np, 'samples')
         sample_exporter.save(data_np, 'data')




if __name__ == '__main__':
  app.run(main)
