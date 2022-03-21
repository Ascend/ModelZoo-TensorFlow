"""
test
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

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.pre_input import get_right_images, r2c, myPSNR, nMse
# import scipy.io as sio
import moxing as mx
from npu_bridge.npu_init import RewriterConfig

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 1, 'Number of samples per batch')
flags.DEFINE_integer('image_size', 256, 'Image sample size in pixels')
flags.DEFINE_integer('random_seed', 0, 'Seed used to initializer rng')
flags.DEFINE_integer('num_epoch', 2000, 'number of epoch')
flags.DEFINE_integer('checkpoint_period', 10, 'save the model every time')
flags.DEFINE_integer('Dn', 11, ' the number of the convolution layers in one residual block')
flags.DEFINE_integer('Dc', 5, 'the number of the data consistency layers')
flags.DEFINE_string('model_name', 'dc', 'model name')
flags.DEFINE_string(
    'train_url', 'obs://imagenet2012-lp/cascade_log/', 'the path of train log in obs')
flags.DEFINE_string('data_url', 'obs://imagenet2012-lp/cascade_re/data/',
                    'the path of train data in obs')
flags.DEFINE_string(
    'data_test_dir', '/home/ma-user/modelarts/inputs/data_url_0/chest_test_acc3.hdf5',
    'the path of train data')
flags.DEFINE_string('last_checkpoint_dir',
                    'obs://imagenet2012-lp/cascade_log/MA-new-cascade_modelarts-11-24-11-26/output',
                    'the path of train data')
flags.DEFINE_string('last_checkpoint_dir_name',
                    '/D10-C7-24-11/', 'the path of train data')
# flags.DEFINE_string('last_checkpoint_file','model-1000','the path of train data')
flags.DEFINE_bool('plot', False, 'show recon results')

saveDir = '/cache/saveModels'
directory = saveDir + FLAGS.last_checkpoint_dir_name
if not os.path.exists(directory):
    os.makedirs(directory)
mx.file.copy_parallel(FLAGS.last_checkpoint_dir + FLAGS.last_checkpoint_dir_name,
                      saveDir + FLAGS.last_checkpoint_dir_name)

tf.reset_default_graph()
feature_tst, label_tst, mask_tst = get_right_images(FLAGS.data_test_dir)

print('Now loading the model ...')
# rec=np.empty(feature_tst.shape,dtype=np.float32)

tf.reset_default_graph()
loadChkPoint = tf.train.latest_checkpoint(directory)
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
reco = []
with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(directory + 'modelTst.meta')
    new_saver.restore(sess, loadChkPoint)
    graph = tf.get_default_graph()
    predT = graph.get_tensor_by_name('predTst:0')
    maskT = graph.get_tensor_by_name('mask:0')
    featureT = graph.get_tensor_by_name('feature:0')

    sess.run(tf.global_variables())
    step = FLAGS.batch_size
    for i in range(0, feature_tst.shape[0], step):
        feature = feature_tst[i:i + step]
        mask = mask_tst[i:i + step]
        dataDict = {featureT: feature, maskT: mask}
        rec = sess.run(predT, feed_dict=dataDict)
        # rec = r2c(rec.squeeze())
        reco.append(rec)

reco = np.concatenate(reco, axis=0)
# sio.savemat('reco.mat',{'reco':reco})

print('Reconstruction done')

print('Now calculating the PSNR (dB) values')

# normOrg=normalize01( np.abs(r2c(label_tst[0])))
# normfeature=normalize01( np.abs(r2c(feature_tst[0])))
# normRec=normalize01(np.abs(rec[0]))
psnrAlised, psnrRec, mseAlised, mseRec = [], [], [], []
for i in range(feature_tst.shape[0]):
    normOrg = np.abs(r2c(label_tst[i]))
    normfeature = np.abs(r2c(feature_tst[i]))
    normRec = np.abs(r2c(reco[i]))

    psnrAlised.append(myPSNR(normOrg, normfeature))
    psnrRec.append(myPSNR(normOrg, normRec))
    mseAlised.append(nMse(normOrg, normfeature))
    mseRec.append(nMse(normOrg, normRec))

psnrAlised = np.mean(psnrAlised)
psnrRec = np.mean(psnrRec)
mseAlised = np.mean(mseAlised)
mseRec = np.mean(mseRec)
print('Avg PSNR: IN\t{0:.6f}\tRE\t{1:.6f}'.format(psnrAlised, psnrRec))
print('Avg MSE: IN\t{0:.6f}\tRE\t {1:.6f}'.format(mseAlised, mseRec))

if FLAGS.plot:
    plot = lambda x: plt.imshow(x, cmap=plt.cm.gray)  
    # plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, .8)) 
    plt.clf()
    plt.subplot(131)
    plot(normOrg)
    plt.axis('off')
    plt.title('Original')
    plt.subplot(132)
    plot(normfeature)
    plt.title('Input, PSNR=' + str(psnrAlised.round(3)) + ' dB' + ',mse=' + str(mseAlised.round(3)))
    plt.axis('off')
    plt.subplot(133)
    plot(normRec)
    plt.title('Output, PSNR=' + str(psnrRec.round(3)) + ' dB' + '.mse=' + str(mseRec.round(3)))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=.01)
    plt.show()
