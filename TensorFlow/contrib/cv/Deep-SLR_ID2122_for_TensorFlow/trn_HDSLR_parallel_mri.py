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

"""
Created on Wed Jun 12 12:09:22 2019

@author: apramanik


Migration for Huawei ModelArts finished on 21th Nov, 2021

@contributor: Robert LIU
"""

# import some libraries
import os
import time
import shutil

from npu_bridge.npu_init import RewriterConfig, npu_config_proto, ExponentialUpdateLossScaleManager, \
    NPULossScaleOptimizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import auxiliaryFunctions as sf
import HDSLR as mm
import scipy as sp
from tst_HDSLR_parallel_mri import test

import getopt
import sys

opts, args = getopt.gnu_getopt(sys.argv[1:], 'd:o:s:', ['data_path=', 'output_path=', 'steps=', 'num='])
print(opts)
data_path = opts[0][1]
output_path = opts[1][1]
num_ = int(opts[3][1])     #控制循环的步数，默认可给0
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# [ModelArts-NPU] Mixed precision configurations
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

# [ModelArts-NPU] Overflow detection dump configurations
overflow_dir = os.path.join(output_path, 'overflow')

if os.path.exists(overflow_dir):         #add
    shutil.rmtree(overflow_dir)
    os.mkdir(overflow_dir)
os.mkdir(overflow_dir)
custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(overflow_dir)
custom_op.parameter_map["enable_dump_debug"].b = True
custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")

# --------------------------------------------------------------
# % SET THESE PARAMETERS CAREFULLY

epochs = int(opts[2][1])  # number of training epochs                      #add
savemodNepoch = 50  # save model after every 'savemodNepoch' epochs

batchSize = 1  # size of batch
K = 1  # number of iterations of data consistency and CNNs
nImg = 360  # number of training images

# --------------------------------------------------------------------------SAME
# Generate a meaningful filename to save the trainined models for testing
print('*************************************************')
start_time = time.time()

saveDir = output_path + 'savedModels/'
cwd = os.getcwd()

dir_name = datetime.now().strftime("%d%b_%I%M_") + str(nImg) + 'I_' + str(epochs) + 'E_' + str(
    batchSize) + 'B_' + str(K) + 'K'
directory = saveDir + dir_name

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName = directory + '/model'

# Read multi-channel dataset
org, atb, mask = sf.getData(data_path, nImg=nImg)

# Save test model
tf.reset_default_graph()
atbT = tf.placeholder(tf.float32, shape=(None, 24, 256, 232, 1), name='atb')
maskT = tf.placeholder(tf.complex64, shape=(None, 12, 256, 232, 1), name='mask')
outk = mm.makeModel(atbT, maskT, K)
fhatT = outk['dc' + str(K)]
fhatT = tf.identity(fhatT, name='fhatT')
sessFileNameTst = directory + '/modelTst'

saver = tf.train.Saver()
with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
    sess.run(tf.global_variables_initializer())
    savedFile = saver.save(sess, sessFileNameTst, latest_filename='checkpointTst')
print('testing model saved:' + savedFile)

# %%
tf.reset_default_graph()
atbP = tf.placeholder(tf.float32, shape=(None, 24, 256, 232, 1), name='atb')
orgP = tf.placeholder(tf.float32, shape=(None, 24, 256, 232, 1), name='org')
maskP = tf.placeholder(tf.complex64, shape=(None, 12, 256, 232, 1), name='mask')

# Creating the dataset
nTrn = org.shape[0]
nBatch = int(np.floor(np.float32(nTrn) / batchSize)) - num_   #控制减少循环步数 5
nSteps = nBatch * epochs  #5*10=50

trnData = tf.data.Dataset.from_tensor_slices((orgP, atbP, maskP))
trnData = trnData.cache()
trnData = trnData.repeat(count=epochs)
trnData = trnData.shuffle(buffer_size=org.shape[0])
trnData = trnData.batch(batchSize, drop_remainder=True)
trnData = trnData.prefetch(5)
iterator = trnData.make_initializable_iterator()
orgT, atbT, maskT = iterator.get_next('getNext')

# Make training model
outk = mm.makeModel(atbT, maskT, K)
fhatT = outk['dc' + str(K)]
fhatT = tf.identity(fhatT, name='fhat')
loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.abs(fhatT - orgT), 2), axis=0))
tf.summary.scalar('lossT', loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# LossScale configurations
opt = tf.train.AdamOptimizer(learning_rate=1e-4)
loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                       decr_every_n_nan_or_inf=2, decr_ratio=0.5)
loss_scale_optimizer = NPULossScaleOptimizer(opt, loss_scale_manager)
optimizer = loss_scale_optimizer.minimize(loss)

# Training code
print('training started at', datetime.now().strftime("%d-%b-%Y %I:%M"))
print('parameters are: Epochs:', epochs, ' BS:', batchSize, 'nSteps:', nSteps, 'nSamples:', nTrn)

saver = tf.train.Saver(max_to_keep=100)
epochloss, totalLoss, ep = [], [], 0
lossT1 = tf.placeholder(tf.float32)
lossSumT = tf.summary.scalar("TrnLoss", lossT1)
perf_list=[]
fps_list=[]

with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
    sess.run(tf.global_variables_initializer())
    feedDict = {orgP: org, atbP: atb, maskP: mask}
    sess.run(iterator.initializer, feed_dict=feedDict)
    savedFile = saver.save(sess, sessFileName)
    print("Model meta graph saved in::%s" % savedFile)

    writer = tf.summary.FileWriter(directory, sess.graph)
    for step in tqdm(range(nSteps)):
        try:
            tmp, _, _ = sess.run([loss, update_ops, optimizer])
            totalLoss.append(tmp)
            if np.remainder(step + 1, nBatch) == 0:
                ep = ep + 1
                step_time = time.time()
                avgTrnLoss = np.mean(totalLoss)
                epochloss.append(avgTrnLoss)
                lossSum = sess.run(lossSumT, feed_dict={lossT1: avgTrnLoss})
                writer.add_summary(lossSum, ep)
                #性能数据获取
                if ep > 2:  #去掉前两次不稳定的数据
                    perf = time.time() - step_time 
                    perf_list.append(perf)
                    perf_mean = np.mean(perf_list)
                    fps = batchSize / perf
                    fps_list.append(fps)
                    fps_mean = np.mean(fps_list)   
                    print("step %d perf %.4f fps %.4f loss %.4f perf_mean %.4f fps_mean %.4f " %(step, perf*1000, fps, avgTrnLoss, perf_mean*1000, fps_mean))
                if np.remainder(ep, savemodNepoch) == 0:
                    savedfile = saver.save(sess, sessFileName, global_step=ep, write_meta_graph=True)
                totalLoss = []
        except tf.errors.OutOfRangeError:
            break
    savedfile = saver.save(sess, sessFileName, global_step=ep, write_meta_graph=True)
    writer.close()
sp.io.savemat(directory + '/epochloss.mat', mdict={'epochs': epochloss}, appendmat=True)
end_time = time.time()
print('Training completed in minutes ', ((end_time - start_time) / 60))
print('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M"))
print('*************************************************')

# Test launch
test(data_path, output_path, directory)
