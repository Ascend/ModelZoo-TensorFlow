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


''' Training Frustum PointNets.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider
from train_util import get_batch

from train_util import get_batch

from cfg import make_config

flags = tf.flags
FLAGS = flags.FLAGS



##paramater
flags.DEFINE_string("result", "result", "The result directory where the model checkpoints will be written.")
flags.DEFINE_string("dataset", "dataset", "dataset path")
flags.DEFINE_string("obs_dir", "obs://fpoint/log", "obs result path, not need on gpu and apulis platform")


## Other parametersresult
flags.DEFINE_string("model","frustum_pointnets_v1","Model name [default: frustum_pointnets_v1]")
flags.DEFINE_string("log_d","log_v1","Log dir [default: log]")
flags.DEFINE_integer("num_point",1024,"Point Number [default: 2048]")
flags.DEFINE_integer("max_epoch",201,"Epoch to run [default: 201]")
flags.DEFINE_integer("batch_size",32,"Batch Size during training [default: 32]")
flags.DEFINE_float("learning_rate",0.001,"Initial learning rate [default: 0.001]")
flags.DEFINE_float("momentum",0.9,"Initial learning rate [default: 0.9]")
flags.DEFINE_string("optimizer","adam","adam or momentum [default: adam]")
flags.DEFINE_integer("decay_step",800000,"Decay step for lr decay [default: 200000]")
flags.DEFINE_float("decay_rate",0.5,"Decay rate for lr decay [default: 0.7]")
flags.DEFINE_boolean("no_intensity",False,"Only use XYZ for training")
flags.DEFINE_string("restore_model_path",None,"Restore model path e.g. log/model.ckpt [default: None]")
flags.DEFINE_string("chip", "npu", "Run on which chip, (npu or gpu or cpu)")
flags.DEFINE_string("platform", "apulis",
                    "Run on apulis/modelarts platform. Modelarts Platform has some extra data copy operations")

## The following params only useful on NPU chip mode
flags.DEFINE_boolean("npu_dump_data", False, "dump data for precision or not")
flags.DEFINE_boolean("npu_dump_graph", False, "dump graph or not")
flags.DEFINE_boolean("npu_profiling", False, "profiling for performance or not")
flags.DEFINE_boolean("npu_auto_tune", False, "auto tune or not. And you must set tune_bank_path param.")


if FLAGS.chip == 'npu':
    from npu_bridge.npu_init import *

# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
# GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL =3 if FLAGS.no_intensity else 4 # point feature channel
NUM_CLASSES = 2 # segmentation has two classes

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = os.path.join(ROOT_DIR, FLAGS.log_dir)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Load Frustum Datasets. Use default data paths.
traindir_1 = os.path.join(FLAGS.dataset, 'frustum_carpedcyc_train.pickle')
validdir_1 = os.path.join(FLAGS.dataset, 'frustum_carpedcyc_val.pickle')
TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='train',
    rotate_to_center=True, random_flip=True, random_shift=True,overwritten_data_path=traindir_1, one_hot=True)
TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
    rotate_to_center=True,overwritten_data_path=validdir_1, one_hot=True)
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    ''' Main function for training and simple evaluation. '''
    with tf.Graph().as_default():
        # with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
        heading_class_label_pl, heading_residual_label_pl, \
        size_class_label_pl, size_residual_label_pl = \
            MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)

        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to increment the 'batch' parameter
        # for you every time it trains.
        batch = tf.get_variable('batch', [],
            initializer=tf.constant_initializer(0), trainable=False)
        bn_decay = get_bn_decay(batch)
        tf.summary.scalar('bn_decay', bn_decay)

        # Get model and losses
        end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
            is_training_pl, bn_decay=bn_decay)
        loss = MODEL.get_loss(labels_pl, centers_pl,
            heading_class_label_pl, heading_residual_label_pl,
            size_class_label_pl, size_residual_label_pl, end_points)
        tf.summary.scalar('loss', loss)

        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        tf.summary.scalar('total_loss', total_loss)

        # Write summaries of bounding box IoU and segmentation accuracies
        iou2ds, iou3ds = tf.py_func(provider.compute_box3d_iou, [\
            end_points['center'], \
            end_points['heading_scores'], end_points['heading_residuals'], \
            end_points['size_scores'], end_points['size_residuals'], \
            centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl], \
            [tf.float32, tf.float32])
        end_points['iou2ds'] = iou2ds
        end_points['iou3ds'] = iou3ds
        tf.summary.scalar('iou_2d', tf.reduce_mean(iou2ds))
        tf.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))

        correct = tf.equal(tf.argmax(end_points['mask_logits'], 2),
            tf.to_int64(labels_pl))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / \
            float(BATCH_SIZE*NUM_POINT)
        tf.summary.scalar('segmentation accuracy', accuracy)

        # Get training operator
        learning_rate = get_learning_rate(batch)
        tf.summary.scalar('learning_rate', learning_rate)
        if OPTIMIZER == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate,
                momentum=MOMENTUM)
        elif OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=batch)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Create a session
        # Create a session for huawei
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        # custom_op.parameter_map["mix_compile_mode"].b =  True
        # custom_op.parameter_map["iterations_per_loop"].i=8
        # custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("GA")
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("ops/ops_info.json")
        # custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(LOG_DIR)
        # custom_op.parameter_map["enable_dump_debug"].b = True
        # custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
        # from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
        #
        # config = tf.ConfigProto()
        # custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        # custom_op.name = "NpuOptimizer"
        # config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap

        # config.gpu_options.allow_growth = True
        # config.allow_soft_placement = True
        # config.log_device_placement = False
        config = make_config(FLAGS)
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if FLAGS.restore_model_path is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)

        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'centers_pred': end_points['center'],
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
    ''' Training for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops
    '''
    is_training = True
    log_string(str(datetime.now()))
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = int(len(TRAIN_DATASET)/BATCH_SIZE)
    #if len(TRAIN_DATASET) % BATCH_SIZE > 0:
    #    num_batches = num_batches + 1

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0

    # Training with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['labels_pl']: batch_label,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training,}

        summary, step, _, loss_val, logits_val, centers_pred_val, \
        iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                ops['logits'], ops['centers_pred'],
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds']], 
                feed_dict=feed_dict)

        train_writer.add_summary(summary, step)

        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt += np.sum(iou3ds>=0.7)

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('segmentation accuracy: %f' % \
                (total_correct / float(total_seen)))
            log_string('box IoU (ground/3D): %f / %f' % \
                (iou2ds_sum / float(BATCH_SIZE*10), iou3ds_sum / float(BATCH_SIZE*10)))
            log_string('box estimation accuracy (IoU=0.7): %f' % \
                (float(iou3d_correct_cnt)/float(BATCH_SIZE*10)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            iou2ds_sum = 0
            iou3ds_sum = 0
            iou3d_correct_cnt = 0
        
        
def eval_one_epoch(sess, ops, test_writer):
    ''' Simple evaluation for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops """
    '''
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = int(len(TEST_DATASET)/BATCH_SIZE)
    #if len(TEST_DATASET) % BATCH_SIZE > 0:
    #    num_batches = num_batches + 1

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0
   
    # Simple evaluation with batches 
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['labels_pl']: batch_label,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training}

        summary, step, loss_val, logits_val, iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['logits'], 
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
                feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label==l)
            total_correct_class[l] += (np.sum((preds_val==l) & (batch_label==l)))
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt += np.sum(iou3ds>=0.7)

        for i in range(BATCH_SIZE):
            segp = preds_val[i,:]
            segl = batch_label[i,:] 
            part_ious = [0.0 for _ in range(NUM_CLASSES)]
            for l in range(NUM_CLASSES):
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): 
                    part_ious[l] = 1.0 # class not present
                else:
                    part_ious[l] = np.sum((segl==l) & (segp==l)) / \
                        float(np.sum((segl==l) | (segp==l)))

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval segmentation accuracy: %f'% \
        (total_correct / float(total_seen)))
    log_string('eval segmentation avg class acc: %f' % \
        (np.mean(np.array(total_correct_class) / \
            np.array(total_seen_class,dtype=np.float))))
    log_string('eval box IoU (ground/3D): %f / %f' % \
        (iou2ds_sum / float(num_batches*BATCH_SIZE), iou3ds_sum / \
            float(num_batches*BATCH_SIZE)))
    log_string('eval box estimation accuracy (IoU=0.7): %f' % \
        (float(iou3d_correct_cnt)/float(num_batches*BATCH_SIZE)))
         
    EPOCH_CNT += 1



if __name__ == "__main__":
    config = make_config(FLAGS)
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
    if FLAGS.platform.lower() == 'modelarts':
        from help_modelarts import modelarts_result2obs
        modelarts_result2obs(FLAGS)
    print("I ran successfully.")