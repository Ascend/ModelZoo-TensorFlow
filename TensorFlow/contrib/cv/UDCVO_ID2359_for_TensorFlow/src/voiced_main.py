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

'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Xiaohan Fei <feixh@cs.ucla.edu>
If you use this code, please cite the following paper:
A. Wong, X. Fei, S. Tsuei, and S. Soatto. Unsupervised Depth Completion from Visual Inertial Odometry.
https://arxiv.org/pdf/1905.08616.pdf
@article{wong2020unsupervised,
  title={Unsupervised Depth Completion From Visual Inertial Odometry},
  author={Wong, Alex and Fei, Xiaohan and Tsuei, Stephanie and Soatto, Stefano},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={1899--1906},
  year={2020},
  publisher={IEEE}
}
'''
from npu_bridge.npu_init import *
import os, time
import numpy as np
import tensorflow as tf
import global_constants as settings
import data_utils
from dataloader import DataLoader
from voiced_model import VOICEDModel
from data_utils import log


def train(train_image_path,
          train_interp_depth_path,
          train_validity_map_path,
          train_intrinsics_path,
          # Batch parameters
          n_batch=settings.N_BATCH,
          n_height=settings.N_HEIGHT,
          n_width=settings.N_WIDTH,
          n_channel=settings.N_CHANNEL,
          # Training settings
          n_epoch=settings.N_EPOCH,
          learning_rates=settings.LEARNING_RATES,
          learning_bounds=settings.LEARNING_BOUNDS,
          # Weights on loss function
          w_ph=settings.W_PH,
          w_co=settings.W_CO,
          w_st=settings.W_ST,
          w_sm=settings.W_SM,
          w_sz=settings.W_SZ,
          w_pc=settings.W_PC,
          # Network settings
          occ_threshold=settings.OCC_THRESHOLD,
          occ_ksize=settings.OCC_KSIZE,
          net_type=settings.NET_TYPE,
          im_filter_pct=settings.IM_FILTER_PCT,
          sz_filter_pct=settings.SZ_FILTER_PCT,
          min_predict_z=settings.MIN_Z,
          max_predict_z=settings.MAX_Z,
          # Pose parameterization
          rot_param=settings.ROT_PARAM,
          pose_norm=settings.POSE_NORM,
          # Model checkpoints and hardware
          n_checkpoint=settings.N_CHECKPOINT,
          n_summary=settings.N_SUMMARY,
          checkpoint_path=settings.CHECKPOINT_PATH,
          restore_path=settings.RESTORE_PATH,
          # Hardware settings
          n_thread=settings.N_THREAD):

  model_path = os.path.join(checkpoint_path, 'model.ckpt')
  log_path = os.path.join(checkpoint_path, 'results.txt')

  # Load image, instrinsics paths from file for training
  train_im_paths = data_utils.read_paths(train_image_path)
  train_iz_paths = data_utils.read_paths(train_interp_depth_path)
  train_vm_paths = data_utils.read_paths(train_validity_map_path)
  train_kin_paths = data_utils.read_paths(train_intrinsics_path)
  assert(len(train_im_paths) == len(train_iz_paths))
  assert(len(train_im_paths) == len(train_vm_paths))
  assert(len(train_im_paths) == len(train_kin_paths))
  n_train_sample = len(train_im_paths)
  n_train_step = n_epoch*np.ceil(n_train_sample/n_batch).astype(np.int32)

  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    # Initialize optimizer
    boundaries = [np.int32((float(v)/n_epoch)*n_train_step) for v in learning_bounds]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, learning_rates)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Add Loss Scale
    loss_scale_opt = optimizer
    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32,
                                                           incr_every_n_steps=1000,
                                                           decr_every_n_nan_or_inf=2,
                                                           decr_ratio=0.5)
    optimizer = NPULossScaleOptimizer(loss_scale_opt, loss_scale_manager)

    # Initialize dataloader
    dataloader = DataLoader(shape=[n_batch, n_height, n_width, n_channel],
                            normalize=True,
                            name='dataloader',
                            n_thread=n_thread,
                            prefetch_size=2*n_thread)
    # Fetch the input from dataloader
    im0 = dataloader.next_element[0]
    im1 = dataloader.next_element[1]
    im2 = dataloader.next_element[2]
    sz0 = dataloader.next_element[3]
    kin = dataloader.next_element[4]

    # Build computation graph
    model = VOICEDModel(im0, im1, im2, sz0, kin,
                        is_training=True,
                        occ_threshold=occ_threshold,
                        occ_ksize=occ_ksize,
                        net_type=net_type,
                        im_filter_pct=im_filter_pct,
                        sz_filter_pct=sz_filter_pct,
                        min_predict_z=min_predict_z,
                        max_predict_z=max_predict_z,
                        rot_param=rot_param,
                        pose_norm=pose_norm,
                        w_ph=w_ph,
                        w_co=w_co,
                        w_st=w_st,
                        w_sm=w_sm,
                        w_sz=w_sz,
                        w_pc=w_pc)
    loss = model.loss
    gradients = optimizer.compute_gradients(loss)
    gradients = optimizer.apply_gradients(gradients, global_step=global_step)

    model_summary = tf.summary.merge_all()

    # Count trainable parameters
    n_parameter = 0
    for variable in tf.trainable_variables():
      n_parameter += np.array(variable.get_shape().as_list()).prod()
    # Log network parameters
    log('Network Parameters:', log_path)
    log('n_batch=%d  n_height=%d  n_width=%d  n_channel=%d' %
        (n_batch, n_height, n_width, n_channel), log_path)
    log('n_sample=%d  n_epoch=%d  n_step=%d  n_param=%d ' %
        (n_train_sample, n_epoch, n_train_step, n_parameter), log_path)
    log('net_type=%s  im_filter_pct=%.3f  sz_filter_pct=%.3f' %
        (net_type, im_filter_pct, sz_filter_pct), log_path)
    log('occ_threshold=%.2f  occ_ksize=%d' %
        (occ_threshold, occ_ksize), log_path)
    log('rot_param=%s  pose_norm=%s' %
        (rot_param, pose_norm), log_path)
    log('min_predict_z=%.3f  max_predict_z=%.3f' %
        (min_predict_z, max_predict_z), log_path)
    log('learning_rates=[%s]' %
        ', '.join('{:.6f}'.format(r) for r in learning_rates), log_path)
    log('boundaries=[%s]' %
        ', '.join('{}:{}'.format(l, v) for l, v in zip(learning_bounds, boundaries)), log_path)
    log('w_ph=%.3f  w_co=%.3f  w_st=%.3f  w_sm=%.3f  w_sz=%.3f  w_pc=%.3f' %
        (w_ph, w_co, w_st, w_sm, w_sz, w_pc), log_path)
    log('Restoring from: %s' % ('None' if restore_path == '' else restore_path),
        log_path)

    # Initialize Tensorflow session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # Add force_fp32
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    # Resolve accuracy issue
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")
    # Resolve preformance issue
    # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    session = tf.Session(config=config)

    # session = tf.Session(config=npu_config_proto(config_proto=config))
    # Initialize saver for storing and restoring checkpoints
    summary_writer = tf.summary.FileWriter(model_path+'-train', session.graph)
    train_saver = tf.train.Saver()
    # Initialize all variables
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    # If given, load the weights from the restore path
    if restore_path != '':
      train_saver.restore(session, restore_path)

    # Begin training
    log('Begin training...', log_path)
    start_step = global_step.eval(session=session)
    time_start = time.time()
    time_batch = time_start
    train_step = start_step

    step = 0
    train_im_paths_epoch, train_iz_paths_epoch, \
        train_vm_paths_epoch, train_kin_paths_epoch = data_utils.make_epoch(
          input_arr=[
            train_im_paths, train_iz_paths, train_vm_paths, train_kin_paths],
          n_batch=n_batch)
    dataloader.initialize(session,
                          image_paths=train_im_paths_epoch,
                          interp_depth_paths=train_iz_paths_epoch,
                          validity_map_paths=train_vm_paths_epoch,
                          intrinsics_paths=train_kin_paths_epoch,
                          do_crop=True)

    while train_step < n_train_step:
      try:
        if train_step % n_summary == 0:
          _, loss_value, summary = session.run([gradients, loss, model_summary])
          summary_writer.add_summary(summary, global_step=train_step)
        else:
          _, loss_value = session.run([gradients, loss])

        if train_step and (train_step % n_checkpoint) == 0:
          time_elapse = (time.time()-time_start)/3600*train_step/(train_step-start_step+1)
          time_remain = (n_train_step/train_step-1)*time_elapse

          #StepTime
          if train_step == n_checkpoint:
            time_step = (time_elapse)*3600/n_checkpoint
          else:
            time_step = (time_elapse-time_batch)*3600/n_checkpoint
          time_batch = time_elapse
          
          #checkpoint_log = 'batch {:>6}  loss: {:.5f}  time elapsed: {:.2f}h  time left: {:.2f}h'
          checkpoint_log = 'batch {:>6}  loss: {:.5f}  time elapsed: {:.2f}h  time left: {:.2f}h  StepTime: {:.6f}s'
          #log(checkpoint_log.format(train_step, loss_value, time_elapse, time_remain), log_path)
          log(checkpoint_log.format(train_step, loss_value, time_elapse, time_remain, time_step), log_path)

          train_saver.save(session, model_path, global_step=train_step)

        train_step += 1
        step += 1
      except tf.errors.OutOfRangeError:
        step = 0
        train_im_paths_epoch, train_iz_paths_epoch, \
            train_vm_paths_epoch, train_kin_paths_epoch = data_utils.make_epoch(
              input_arr=[
                train_im_paths, train_iz_paths, train_vm_paths, train_kin_paths],
              n_batch=n_batch)
        dataloader.initialize(session,
                              image_paths=train_im_paths_epoch,
                              interp_depth_paths=train_iz_paths_epoch,
                              validity_map_paths=train_vm_paths_epoch,
                              intrinsics_paths=train_kin_paths_epoch,
                              do_crop=True)

    train_saver.save(session, model_path, global_step=n_train_step)

