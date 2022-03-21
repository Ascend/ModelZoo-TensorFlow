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
import sys
from tensorflow.contrib import layers
from sklearn import svm, manifold
# from mayavi import mlab
import os
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# 导入网络模型文件
from vsl import VarShapeLearner

# define network structure, parameters
global_latent_dim  = 20
local_latent_dim   = 10
local_latent_num   = 5
obj_res     = 30
batch_size  = 200
print_step  = 1
total_epoch = 200

# 3D visualization
def draw_sample(voxel,  savepath):
    voxel = np.reshape(voxel, (obj_res, obj_res, obj_res))
    xx, yy, zz = np.where(voxel >= 0)
    ss = voxel[np.where(voxel >= 0)] * 1.
    # mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(400, 400))
    # s = mlab.points3d(xx, yy, zz, ss,
    #                   mode="cube",
    #                   colormap='bone',
    #                   scale_factor=2)
    # mlab.view(120, 290, 85)
    
    # s.scene.light_manager.lights[0].activate  = True
    # s.scene.light_manager.lights[0].intensity = 1.0
    # s.scene.light_manager.lights[0].elevation = 30
    # s.scene.light_manager.lights[0].azimuth   = -30
    # s.scene.light_manager.lights[1].activate  = True
    # s.scene.light_manager.lights[1].intensity = 0.3
    # s.scene.light_manager.lights[1].elevation = -60
    # s.scene.light_manager.lights[1].azimuth   = -30
    # s.scene.light_manager.lights[2].activate  = False
    # if savepath == 0:
    #     return mlab.show()
    # return  mlab.savefig(savepath)

# load dataset (pick modelnet40 or modelnet10)
data = h5py.File('/cache/modelnet40/ModelNet40_res30_raw.mat')

train_all = np.transpose(data['train'])
test_all  = np.transpose(data['test'])


# load VSL model
VSL = VarShapeLearner(obj_res=obj_res,
                      batch_size=batch_size,
                      global_latent_dim=global_latent_dim,
                      local_latent_dim=local_latent_dim,
                      local_latent_num=local_latent_num)

# load saved parameters here, comment this to train model from scratch.
# VSL.saver.restore(VSL.sess, os.path.abspath('parameters/modelnet40-2619-cost-1.1170.ckpt'))


# training VSL model
for epoch in range(total_epoch):
    cost     = np.zeros(3, dtype=np.float32)
    avg_cost = np.zeros(3, dtype=np.float32)
    train_batch = int(train_all.shape[0] / batch_size)

    index = epoch + 0 # correct the training index, set 0 for training from scratch

    # iterate for all batches
    np.random.shuffle(train_all)
    for i in range(train_batch):
        x_train = train_all[batch_size*i:batch_size*(i+1),1:].reshape([batch_size, obj_res, obj_res, obj_res, 1])

        # calculate and average kl and vae loss for each batch
        cost[0] = np.mean(VSL.sess.run(VSL.kl_loss_all, feed_dict={VSL.x: x_train}))
        cost[1] = np.mean(VSL.sess.run(VSL.rec_loss, feed_dict={VSL.x: x_train}))
        cost[2] = VSL.model_fit(x_train)
        avg_cost += cost / train_batch

    print("Epoch: {:04d} | kl-loss: {:.4f} + rec-loss: {:.4f} = total-loss: {:.4f}"
          .format(index, avg_cost[0], avg_cost[1], avg_cost[2]))

    if epoch % print_step == 0:
        # draw_sample(VSL.sess.run(VSL.x_rec[0,:], feed_dict={VSL.x: x_train}), 'plots/rec-%d.png' % index)
        # mlab.close()

        VSL.saver.save(VSL.sess, os.path.abspath('/cache/result/modelnet40-{:04d}-cost-{:.4f}.ckpt'.format(index, avg_cost[2])))


# shape classification using SVM
'''note: this process will concatenate all features in the dataset
which will be needed for tsne output.'''

train_batch = int(train_all.shape[0] / batch_size)
np.random.shuffle(train_all)
for i in range(train_batch):
    x_train = train_all[batch_size * i:batch_size * (i + 1), 1:].reshape([batch_size, obj_res, obj_res, obj_res, 1])
    if i == 0:
        train_feature = VSL.sess.run(VSL.latent_feature, feed_dict={VSL. x:x_train})
    else:
        train_feature = np.concatenate([train_feature, VSL.sess.run(VSL.latent_feature, feed_dict={VSL. x:x_train})])

np.random.shuffle(test_all)
test_batch = int(test_all.shape[0] / batch_size)
for i in range(test_batch):
    x_test = test_all[batch_size * i:batch_size * (i + 1), 1:].reshape(
        [batch_size, obj_res, obj_res, obj_res, 1])
    if i == 0:
        test_feature = VSL.sess.run(VSL.latent_feature, feed_dict={VSL.x: x_test})
    else:
        test_feature = np.concatenate(
            [test_feature, VSL.sess.run(VSL.latent_feature, feed_dict={VSL.x: x_test})])

clf = svm.SVC(kernel='rbf')
clf.fit(train_feature[:,:], train_all[0:batch_size * train_batch, 0])

train_accuracy = np.sum(train_all[0:batch_size * train_batch, 0] == clf.predict(train_feature[:,:])) / (train_batch * batch_size)
test_accuracy  = np.sum(test_all[0:batch_size * test_batch, 0] == clf.predict(test_feature[:,:])) / (test_batch * batch_size)

print('Shape classification: train: {:.4f}, test: {:.4f}'
      .format(train_accuracy, test_accuracy))


# t-sne 2D visualization
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(train_feature)
ax = plt.figure(figsize=(8, 8), facecolor='white')
plt.scatter(Y[:, 0], Y[:, 1], c=train_all[0:batch_size * train_batch, 0], edgecolors='none', cmap='terrain')
plt.xticks([])
plt.yticks([])
plt.axis('tight')
plt.show()
plt.savefig('/cache/result/tsne.png')


# shape generation with Gaussian noise
'''please fetch model id number from ModelNet: http://modelnet.cs.princeton.edu/
ModelNet40 and ModelNet10 has 40 and 10 classes respectively with alphabetical naming order.
Here is the example: id = 1 means with all models in "airplane" class. '''
id = 1
test_all  = np.transpose(data['test'])
test_indx  = np.where(test_all[:,0] == id)
test  = test_all[test_indx[-0],1:]
x_test = test_all[test_indx[0][0]:test_indx[0][0]+batch_size,1:].reshape([batch_size, obj_res, obj_res, obj_res, 1])

z = VSL.sess.run(VSL.latent_feature, feed_dict={VSL.x: x_test})
z_new = z + np.random.normal(scale=0.02, size=[batch_size, local_latent_dim*local_latent_num+global_latent_dim])
# for i in range(20):
    # draw_sample(VSL.sess.run(VSL.x_rec[i, :], feed_dict={VSL.latent_feature: z_new}), 'plots/rec_{:d}.png'.format(i))
#     mlab.close()


# shape interpolation
'''shape interpolation visualization from two reconstructed shapes.
id1 and id2 means two shape instances in the reconstructed shape batches.
note: id number cannot exceed the batch_size.'''
z = VSL.sess.run(VSL.latent_feature, feed_dict={VSL.x: x_test})
id1 = 8
id2 = 0
d = z[id1, :] - z[id2, :]
for i in range(7):
    # draw_sample(VSL.sess.run(VSL.x_rec[id2, :], feed_dict={VSL.latent_feature: z}), 'plots/interpolation_airplane-{:d}.png'.format(i))
    # mlab.close()
    z[id2, :] = z[id2, :] + d / 6


