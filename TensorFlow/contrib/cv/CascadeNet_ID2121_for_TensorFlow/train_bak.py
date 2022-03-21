"""
train
"""
# coding=utf-8
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

import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
from datetime import datetime
import os
import time
from models.pre_input import get_right_images
import models.model_tf as mm
import moxing as mx
from npu_bridge.npu_init import RewriterConfig
# if not work, please use import *
# from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig



flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 16, 'Number of samples per batch')
flags.DEFINE_integer('image_size', 256, 'Image sample size in pixels')
flags.DEFINE_integer('random_seed', 0, 'Seed used to initializer rng')
flags.DEFINE_integer('num_epoch', 500, 'number of epoch')
flags.DEFINE_integer('checkpoint_period', 10, 'save the model every time')
flags.DEFINE_integer(
    'Dn', 10, ' the number of the convolution layers in one residual block')
flags.DEFINE_integer('Dc', 7, 'the number of the data consistency layers')
flags.DEFINE_string('model_name', 'dc', 'model name')
flags.DEFINE_string('data_url', 'obs://imagenet2012-lp/cascade_re/data/',
                    'the path of train data in obs')
flags.DEFINE_string(
    'data_train_dir', '/home/ma-user/modelarts/inputs/data_url_0/chest_train_acc3.hdf5',
    'the path of train data')
flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate')
flags.DEFINE_bool('continue_training', False, 'continue training')
flags.DEFINE_string(
    'train_url', 'obs://imagenet2012-lp/cascade_log/', 'the path of train log in obs')
flags.DEFINE_string('last_checkpoint_dir',
                    'obs://imagenet2012-lp/cascade_log/MA-cascade_modelarts-10-19-15-26/output/V0018',
                    'the path of train data')
flags.DEFINE_string('last_checkpoint_dir_name',
                    '/D11-C5-25-19/', 'the path of train data')

print('***************************************************')
start_time = time.time()
# creat checkpoint save path
saveDir = '/cache/saveModels'
cwd = os.getcwd()
directory = saveDir + '/' + 'D' + \
            str(FLAGS.Dn) + '-C' + str(FLAGS.Dc) + \
            '-' + datetime.now().strftime("%d-%H")
if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName = directory + '/model'
image_size = FLAGS.image_size
# net architecture
K = FLAGS.Dc
numlayers = FLAGS.Dn

tf.reset_default_graph()
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
# set precision mode allow_fp32_to_fp16  allow_mix_precision
custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes(
    'allow_fp32_to_fp16')
# # dump path
# custom_op.parameter_map['dump_path'].s = tf.compat.as_bytes(saveDir + '/')
# # set dump debug
# custom_op.parameter_map['enable_dump_debug'].b = True
# custom_op.parameter_map['dump_debug_mode'].s = tf.compat.as_bytes('all')
# custom_op.parameter_map["profiling_mode"].b = True
# custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
#     '{"output":"/cache/saveModels","task_trace":"on"}')
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # have to close
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # have to close

# save model
tf.reset_default_graph()
mask = tf.placeholder(tf.float32, shape=(
    None, image_size, image_size), name='mask')
feature = tf.placeholder(tf.float32, shape=(
    None, image_size, image_size, 2), name='feature')

out = mm.makeModel(feature, mask, train=False, nLayers=numlayers, K=K)
predTst = out['dc' + str(K)]
predTst = tf.identity(predTst, name='predTst')
sessFileNameTst = directory + '/modelTst'

saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saveFile = saver.save(sess, sessFileNameTst,
                          latest_filename='checkpointTst')
print('testing model saved:' + saveFile)

# read dataset
# mx.file.copy_parallel(FLAGS.data_url, '/cache/data/')  # copy to modelarts
path_train = FLAGS.data_train_dir
feature_trn, label_trn, mask_trn = get_right_images(path_train)
if FLAGS.continue_training:
    mx.file.copy_parallel(FLAGS.last_checkpoint_dir + FLAGS.last_checkpoint_dir_name,
                          saveDir + FLAGS.last_checkpoint_dir_name)

tf.reset_default_graph()
rows = image_size
cols = image_size
channels = 2
input_feature = tf.placeholder(
    tf.float32, shape=[None, rows, cols, channels], name='feature')
input_label = tf.placeholder(
    tf.float32, shape=[None, rows, cols, channels], name='label')
input_mask = tf.placeholder(tf.float32, shape=[None, rows, cols], name='mask')
nTrn = int(feature_trn.shape[0])
nBatch = int(np.floor(np.float32(nTrn) / FLAGS.batch_size))
nSteps = nBatch * FLAGS.num_epoch

trnData = tf.data.Dataset.from_tensor_slices(
    (input_feature, input_label, input_mask))
trnData = trnData.repeat(count=FLAGS.num_epoch)
trnData = trnData.shuffle(buffer_size=nTrn)
trnData = trnData.batch(batch_size=FLAGS.batch_size, drop_remainder=True)
trnData = trnData.prefetch(20)
# iterator = trnData.make_initializable_iterator()
iterator = tf.data.make_initializable_iterator(trnData)
feature_t, label_t, mask_t = iterator.get_next('getnext')

# loss function
out = mm.makeModel(feature_t, mask_t, True, numlayers, K)
recon = out['dc' + str(K)]
recon = tf.identity(recon, name='pred')
mse_loss = tf.losses.mean_squared_error(recon, label_t)
tf.add_to_collection('losses', mse_loss)
loss = tf.add_n(tf.get_collection('losses'))

global_step = tf.Variable(
    0, dtype=tf.int32, trainable=False, name="global_step")
# opti = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate,
#                               name='optimizer')
decayed_lr = tf.train.exponential_decay(
    FLAGS.learning_rate, global_step, 1000, 0.98, staircase=True)
# opti = tf.train.AdamOptimizer(learning_rate=decayed_lr, name='optimizer')
opti = tf.train.GradientDescentOptimizer(decayed_lr,
                                         name='optimizer')
# loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32,
# incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
# opti = NPULossScaleOptimizer(opt, loss_scale_manager)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
# train_op = optimizer.minimize(loss)
optimizer = opti.minimize(loss, global_step)

print('training started at', datetime.now().strftime('%d%b_%I:%M%p'))
print('parameters are: Epoch:', FLAGS.num_epoch, 'BS:',
      FLAGS.batch_size, 'nSteps:', nSteps, 'nSamples:', nTrn)

saver = tf.train.Saver(max_to_keep=100)
totalLoss, ep = [], 0
lossT = tf.placeholder(tf.float32)
lossSumT = tf.summary.scalar('TrnLoss', lossT)
merged = tf.summary.merge_all()
# sess
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if FLAGS.continue_training:
        ckpt = tf.train.get_checkpoint_state(
            saveDir + FLAGS.last_checkpoint_dir_name)
        print(ckpt)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('restore the model from {}.'.format(
                FLAGS.last_checkpoint_dir + FLAGS.last_checkpoint_dir_name))
        else:
            pass
    feedDict = {input_feature: feature_trn,
                input_label: label_trn, input_mask: mask_trn}
    sess.run(iterator.initializer, feed_dict=feedDict)
    savedFile = saver.save(sess, sessFileName)

    print('Model meta graph saved in ::%s' % savedFile)
    writer = tf.summary.FileWriter(directory, sess.graph)
    saveLoss = []
    # for step in tqdm(range(nSteps)):
    for step in range(nSteps):
        try:
            tmp, _, _ = sess.run([mse_loss, update_ops, optimizer])
            totalLoss.append(tmp)
            if np.remainder(step + 1, nBatch) == 0:
                ep += 1
                avgTrnLoss = np.mean(totalLoss) / nTrn
                summary = sess.run(merged, feed_dict={lossT: avgTrnLoss})
                writer.add_summary(summary, ep)
                saveLoss.append(avgTrnLoss)
                totalLoss = []
                print(datetime.now().strftime("%H:%M"),
                      '---Epoch: ', ep, '---AvgLoss: ', avgTrnLoss)
                # todo
                if np.remainder(ep, FLAGS.checkpoint_period) == 0:
                    savedfile = saver.save(
                        sess, sessFileName, global_step=ep, write_meta_graph=True)
        except tf.errors.OutOfRangeError:
            break
    savedfile = saver.save(
        sess, sessFileName, global_step=ep, write_meta_graph=True)
    writer.close()
end_time = time.time()
print('Training completed in minutes', ((end_time - start_time) / 60))
print('training completed at', datetime.now().strftime('%d-%b-%Y %I:%M%p'))
print('****************************************************')
# copy results to obs
mx.file.copy_parallel('/cache/saveModels', FLAGS.train_url)
print('copy saved model to obs.')
