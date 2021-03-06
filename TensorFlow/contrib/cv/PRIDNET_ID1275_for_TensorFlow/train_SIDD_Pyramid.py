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

import h5py
import glob
import os, time
import tensorflow as tf
import numpy as np
from network import network
from PIL import Image
#from help_modelarts import modelarts_result2obs

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags = tf.flags
Flags = flags.FLAGS

flags.DEFINE_string("result", "result", "train model result")
flags.DEFINE_string("dataset", "dataset", "train model dataset url")
flags.DEFINE_string("train_model_url", "train_model_url", "save model checkpoint url")
flags.DEFINE_string("obs_dir", "obs_dir", "obs url")
flags.DEFINE_string("chip", "cpu", "train platform")
flags.DEFINE_integer("train_step", 100, "model train step")
flags.DEFINE_boolean("profiling", False, "model profiling start")

dir_name = Flags.dataset
checkpoint_dir = Flags.train_model_url
result_dir = Flags.result

file_list = glob.glob(dir_name + '/*/*NOISY_RAW_010*')
gt_list = glob.glob(dir_name + '/*/*GT_RAW_010*')

train_ids = [os.path.basename(train_fn)[0:4] for train_fn in file_list]

mat_img = {}
gt_img = {}
start = time.time()
index = 0
for file, gt_file in zip(file_list, gt_list):
    key = os.path.basename(file)[0:4]
    file_1 = file[:-5] + '1.MAT'
    gt_file_1 = gt_file[:-5] + '1.MAT'

    index = index + 1
    print(index, 'loading file: ', key)
    m = h5py.File(file, 'a')['x']
    m = np.expand_dims(np.expand_dims(m, 0), 3)
    m_1 = h5py.File(file_1, 'a')['x']
    m_1 = np.expand_dims(np.expand_dims(m_1, 0), 3)
    mat_img[key] = np.concatenate([m, m_1], 0)

    m_gt = h5py.File(gt_file, 'a')['x']
    m_gt = np.expand_dims(np.expand_dims(m_gt, 0), 3)
    m_gt_1 = h5py.File(gt_file_1, 'a')['x']
    m_gt_1 = np.expand_dims(np.expand_dims(m_gt_1, 0), 3)
    gt_img[key] = np.concatenate([m_gt, m_gt_1], 0)

ps = 256  # patch size for training
save_freq = 500

if Flags.chip == "npu":
    from npu_bridge.npu_init import *
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True  # ??????????????????????????????AI?????????????????????
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # ??????????????????
    # profiling configuration
    if Flags.profiling:
        work_dir = os.getcwd()
        profiling_dir = os.path.join(work_dir, "npu_profiling")
        if not os.path.exists(profiling_dir):
            os.makedirs(profiling_dir)
        options = '{"output":"%s", "task_trace":"on", "aicpu":"on"}' % profiling_dir
        custom_op.parameter_map["profiling_mode"].b = True
        custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(options)
    sess = tf.Session(config=config)
else:
    sess = tf.Session()

in_image = tf.placeholder(tf.float32, [None, None, None, 1])
gt_image = tf.placeholder(tf.float32, [None, None, None, 1])

out_image = network(in_image)

# h_tv = tf.nn.l2_loss(feature_map[:, 1:, :, :] - feature_map[:, :-1, :, :])
# w_tv = tf.nn.l2_loss(feature_map[:, :, 1:, :] - feature_map[:, :, :-1, :])
# tv_loss = (h_tv + w_tv) / (255 * 256)
G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
# G_loss = G_loss_2 + 0.1 * tv_loss


tf.summary.scalar('G_loss', G_loss)
merged = tf.summary.merge_all()

t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)

#G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)
G_opt = tf.train.AdamOptimizer(learning_rate=lr)
G_opt = tf.train.experimental.MixedPrecisionLossScaleOptimizer(G_opt, loss_scale=2 ** 32).minimize(G_loss)

saver = tf.train.Saver(max_to_keep=15)
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

g_loss = np.zeros((5000, 1))

allfolders = glob.glob(result_dir + '/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

summary_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)

learning_rate = 1e-4

epoch_loss_list = []
min_epoch_loss = 50
for epoch in range(lastepoch, Flags.train_step):
    if os.path.isdir("result/%04d" % epoch):
        continue
    if epoch > 1500:
        learning_rate = 5e-5
    if epoch > 2000:
        learning_rate = 1e-5
    if epoch > 2500:
        learning_rate = 5e-6
    if epoch > 3000:
        learning_rate = 1e-6
    if epoch > 3500:
        learning_rate = 5e-7

    cnt = 0
    epoch_loss = 0

    for ind in np.random.permutation(len(train_ids)):

        st = time.time()
        cnt += 1

        train_id = train_ids[ind]  # string
        train_batch = mat_img[train_id]
        gt_batch = gt_img[train_id]

        # crop
        H = train_batch.shape[1]
        W = train_batch.shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = train_batch[:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_batch[:, yy:yy + ps, xx:xx + ps, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        _, G_current, output, summary = sess.run([G_opt, G_loss, out_image, merged],
                                                 feed_dict={in_image: input_patch, gt_image: gt_patch,
                                                            lr: learning_rate})

        output = np.minimum(np.maximum(output, 0), 1)
        g_loss[ind] = G_current
        epoch_loss += G_current
        summary_writer.add_summary(summary, cnt + epoch * len(train_ids))

        print("%d %d Loss = %.4f Time = %.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((gt_patch[0, :, :, 0], output[0, :, :, 0]), axis=1)
            Image.fromarray((temp * 255).astype('uint8'), mode='L').convert('RGB').save(
                result_dir + '%04d/%04d_00_train.jpg' % (epoch, int(train_id)))

    epoch_loss /= len(train_ids)
    epoch_loss_list.append(epoch_loss)
    epoch_summary = tf.Summary(value=[tf.Summary.Value(tag='epoch_loss', simple_value=epoch_loss)])
    summary_writer.add_summary(summary=epoch_summary, global_step=epoch)
    summary_writer.flush()

    if epoch_loss_list[epoch] < min_epoch_loss:
        saver.save(sess, checkpoint_dir + 'model.ckpt')
        with open(checkpoint_dir + 'log.txt', 'a+') as log:
            log.write('saved epoch: %04d, epoch loss = ' % epoch + str(epoch_loss) + '\n')
        print('saved epoch: %04d' % epoch)
        print(epoch_loss)
        min_epoch_loss = epoch_loss_list[epoch]
    if epoch >= 3990:
        saver.save(sess, checkpoint_dir + 'model-%04d.ckpt' % epoch)
        with open(checkpoint_dir + 'log.txt', 'a+') as log:
            log.write('final saved epoch: %04d, epoch loss = ' % epoch + str(epoch_loss) + '\n')

# copy dataset from modelarts to obs
#modelarts_result2obs(Flags)
sess.close()
