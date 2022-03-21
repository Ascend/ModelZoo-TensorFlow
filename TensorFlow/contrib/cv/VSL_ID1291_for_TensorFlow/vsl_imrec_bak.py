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

from npu_bridge.npu_init import *
from tensorflow.contrib import layers
# from mayavi import mlab
import os
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# 导入网络模型
from vsl_rec import VarShapeLearner


# define network structure, parameters
global_latent_dim  = 5
local_latent_dim   = 2
local_latent_num   = 3
obj_res     = 30
batch_size  = 5
print_step  = 2
total_epoch = 10

# 3D visualization
def draw_sample(voxel,  savepath):
    voxel = np.reshape(voxel, (obj_res, obj_res, obj_res))
    xx, yy, zz = np.where(voxel >= 0)
    ss = voxel[np.where(voxel >= 0)] * 1.
    mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(400, 400))
    s = mlab.points3d(xx, yy, zz, ss,
                      mode="cube",
                      colormap='bone',
                      scale_factor=2)
    mlab.view(112, 242, 80)
    # s.scene.light_manager.lights[0].activate  = True
    # s.scene.light_manager.lights[0].intensity = 1.0
    # s.scene.light_manager.lights[0].elevation = 50
    # s.scene.light_manager.lights[0].azimuth   = -30
    # s.scene.light_manager.lights[1].activate  = True
    # s.scene.light_manager.lights[1].intensity = 0.3
    # s.scene.light_manager.lights[1].elevation = -40
    # s.scene.light_manager.lights[1].azimuth   = -30
    # s.scene.light_manager.lights[2].activate  = False
    if savepath == 0:
        return mlab.show()
    return  mlab.savefig(savepath)


# load dataset (only PASCAL3D in this case)
PASCAL = h5py.File('/cache/modelnet40/PASCAL3D.mat')
image_train = np.transpose(PASCAL['image_train'])
model_train = np.transpose(PASCAL['model_train'])
image_test = np.transpose(PASCAL['image_test'])
model_test = np.transpose(PASCAL['model_test'])

# load VSL model
VSL = VarShapeLearner(obj_res=obj_res,
                      batch_size=batch_size,
                      global_latent_dim=global_latent_dim,
                      local_latent_dim=local_latent_dim,
                      local_latent_num=local_latent_num)

# load saved parameters here, comment this to train model from scratch.
# VSL.saver.restore(VSL.sess, os.path.abspath('parameters/your_model_name.ckpt'))

def unison_shuffled_copies(a, b):
    '''solution using: 
    http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison/4602224'''
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def IOU(a,b):
    index = np.argwhere(a == 1)
    index_gt = np.argwhere(b == 1)
    intersect = np.intersect1d(index, index_gt)
    union = np.union1d(index, index_gt)
    IOU = len(intersect) / len(union)
    return IOU

# training VSL model
name_list = ['aero', 'bike', 'boat', 'bus', 'car', 'chair', 'mbike', 'sofa', 'train', 'tv']

id = 1 # training separately per class, using id from the name_list [1:10]
test_indx  = np.where(model_test[:,0] == id)
modelid_test = model_test[test_indx[0], 1:]
imageid_test = image_test[test_indx[0],:]

train_indx  = np.where(model_train[:,0] == id)
modelid_train = model_train[train_indx[0], 1:]
imageid_train = image_train[train_indx[0],:]

for epoch in range(total_epoch):
    cost     = np.zeros(4, dtype=np.float32)
    avg_cost = np.zeros(4, dtype=np.float32)
    train_batch = int(imageid_train.shape[0] / batch_size)

    index = epoch + 0  # correct the training index, set 0 for training from scratch

    # randomly shuffle for each epoch
    [imageid_train, modelid_train] = unison_shuffled_copies(imageid_train, modelid_train)

    # warming-up schedule
    if index <= 50:
        gamma = 10 ** (np.floor(index / 10) - 8)
    elif 50 < index < 100:
        gamma = np.floor((index - 40) / 10) * 10 ** (-3)
    else:
        gamma = 5 * 10 ** (-3)

    # iterate for all batches
    for i in range(train_batch):
        x_train = modelid_train[batch_size*i:batch_size*(i+1),:].reshape([batch_size, obj_res, obj_res, obj_res, 1])
        y_train = imageid_train[batch_size*i:batch_size*(i+1),:]

        # calculate and average kl, rec and latent loss for each batch
        cost[0] = np.mean(VSL.sess.run(VSL.kl_loss_all, feed_dict={VSL.x: x_train, VSL.y: y_train,
                                                                   VSL.gamma: gamma, VSL.keep_prob: 0.2}))
        cost[1] = np.mean(VSL.sess.run(VSL.rec_loss, feed_dict={VSL.x: x_train, VSL.y: y_train,
                                                                VSL.gamma: gamma, VSL.keep_prob: 0.2}))
        cost[2] = np.mean(VSL.sess.run(VSL.lat_loss, feed_dict={VSL.x: x_train, VSL.y: y_train,
                                                                VSL.gamma: gamma, VSL.keep_prob: 0.2}))
        cost[3] = VSL.model_fit(x_train, y_train, gamma, 0.2)

        avg_cost += cost / train_batch

    print("Epoch: {:04d} | kl-loss: {:.4f} + rec-loss: {:.4f} + lat_loss: {:.4f} = total-loss: {:.4f}"
          .format(index, avg_cost[0], avg_cost[1], avg_cost[2], avg_cost[3]))

    if index % print_step == 0:
        # draw_sample(VSL.sess.run(VSL.x_rec[0,:], feed_dict={VSL.x: x_train, VSL.y: y_train, VSL.gamma: gamma}), 'plots/rec-%d.png' % index)
        # mlab.close()

        VSL.saver.save(VSL.sess, os.path.abspath('/cache/result/{}-{:03d}-3-2-5-cost-{:.4f}.ckpt'
                                                 .format(name_list[id-1], index, avg_cost[3])))

# IOU training and testing results
test_batch = int(modelid_test.shape[0] / batch_size)
z_train = [[0]]*test_batch
z_test = [[0]]*test_batch
for i in range(test_batch):
    x_train = modelid_train[batch_size * i:batch_size * (i + 1), :].reshape([batch_size, obj_res, obj_res, obj_res, 1])
    y_train = imageid_train[batch_size * i:batch_size * (i + 1), :]
    x_test = modelid_test[batch_size * i:batch_size * (i + 1), :].reshape([batch_size, obj_res, obj_res, obj_res, 1])
    y_test = imageid_test[batch_size * i:batch_size * (i + 1), :]

    z_train[i] = VSL.sess.run(VSL.learned_feature, feed_dict={VSL.x: x_train, VSL.y: y_train, VSL.gamma: gamma, VSL.keep_prob:1})
    z_test[i] = VSL.sess.run(VSL.learned_feature, feed_dict={VSL.x: x_test, VSL.y: y_test, VSL.gamma: gamma, VSL.keep_prob:1})

    if i == 0:
        train_rec = VSL.sess.run(VSL.x_rec, feed_dict={VSL.latent_feature: z_train[i]})
        test_rec = VSL.sess.run(VSL.x_rec, feed_dict={VSL.latent_feature: z_test[i]})
    else:
        train_rec = np.concatenate((train_rec,  VSL.sess.run(VSL.x_rec, feed_dict={VSL.latent_feature: z_train[i]})))
        test_rec = np.concatenate((test_rec,  VSL.sess.run(VSL.x_rec, feed_dict={VSL.latent_feature: z_test[i]})))

train_rec = np.floor(train_rec + 0.5)
train_rec = train_rec.reshape(len(train_rec), obj_res ** 3)

test_rec = np.floor(test_rec + 0.5)
test_rec = test_rec.reshape(len(test_rec), obj_res ** 3)

prob_train = 0
prob_test = 0
test_batch = int(modelid_test.shape[0] / batch_size)
for i in range(batch_size * test_batch):
    prob_model = IOU(train_rec[i, :], modelid_train[i, :])
    prob_train = prob_model / (batch_size * test_batch) + prob_train

    prob_model = IOU(test_rec[i, :], modelid_test[i, :])
    prob_test = prob_model / (batch_size * test_batch) + prob_test

print('IOU - {} - Train: {:.4f}, Test: {:.4f}'.format(name_list[id-1], prob_train, prob_test))


# image reconstruction
test_indx  = np.where(model_test[:,0] == id)
modelid_test = model_test[test_indx[0], 1:]
imageid_test = image_test[test_indx[0],:]
test_batch = int(modelid_test.shape[0] / batch_size)
gamma = 5e-3
z_learned = [[0]]*test_batch
for i in range(test_batch):
    x_train = modelid_test[batch_size * i:batch_size * (i + 1), :].reshape([batch_size, obj_res, obj_res, obj_res, 1])
    y_train = imageid_test[batch_size * i:batch_size * (i + 1), :]

    z_learned[i] = VSL.sess.run(VSL.learned_feature, feed_dict={VSL.x: x_train, VSL.y: y_train, VSL.gamma: gamma, VSL.keep_prob:1})
    if i == 0:
        A = VSL.sess.run(VSL.x_rec, feed_dict={VSL.latent_feature: z_learned[i]})
    else:
        A = np.concatenate((A,  VSL.sess.run(VSL.x_rec, feed_dict={VSL.latent_feature: z_learned[i]})))

A = np.floor(A + 0.5)
A = A.reshape(len(A), obj_res ** 3)

# plot image and its 3d reconstructed model
for i in range(20):
    plt.imshow(imageid_test[i, :])
    plt.axis('off')
    plt.savefig('/cache/result/{}-im{:d}.png'.format(name_list[id-1], i))
    plt.close()
    # draw_sample(A[i, :], 'plots/{}-md{:d}.png'.format(name_list[id-1], i))
    # mlab.close()



