#
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
import argparse
import importlib
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataset.util.tfrecord import get_dataset_generator
from dataset.util.object_detection import preprocess
from model.centernet import CenterNet, loss


def main(platform_cfg, dataset_cfg, task_cfg):
    """test centernet on VOC

    Args:
        platform_cfg: config.platform.?
        dataset_cfg: config.dataset.?
        task_cfg: config.task.?
    """
    tfrecord_paths = [
        os.path.join(platform_cfg.WORKSPACE_DIR, dataset_path) for dataset_path in dataset_cfg.TRAIN_DATASET_PATHS
    ]
    dataset_init, dataset_iter = get_dataset_generator(tfrecord_paths,
                                                       lambda x: preprocess(x, task_cfg.AUGMENTOR_CONFIG),
                                                       shuffle_buffer_size=256,
                                                       batch_size=task_cfg.BATCH_SIZE,
                                                       drop_remainder=True,
                                                       prefetch_buffer_size=2)

    _, img, _, ground_truth = dataset_iter.get_next()
    img.set_shape([task_cfg.BATCH_SIZE, *task_cfg.RESIZE_IMG_SHAPE, 3])
    img_mean = tf.convert_to_tensor(task_cfg.DATA_MEAN, dtype=tf.float32)
    img_mean = tf.reshape(img_mean, [1, 1, 1, 3])
    img_std = tf.convert_to_tensor(task_cfg.DATA_STD, dtype=tf.float32)
    img_std = tf.reshape(img_std, [1, 1, 1, 3])
    img = (img / 255. - img_mean) / img_std

    heatmap, size, offset = CenterNet(is_training=task_cfg.IS_TRAINING, num_classes=dataset_cfg.NUM_CLASSES).call(img)

    heatmap_loss, size_loss, offset_loss, total_loss = loss(heatmap, size, offset, ground_truth, task_cfg.BATCH_SIZE,
                                                            dataset_cfg.NUM_CLASSES)

    with tf.name_scope('train_op'):
        # lr = tf.placeholder(dtype=tf.float32, shape=(1), name='lr')
        lr = 1e-3
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(total_loss)

    saver = tf.train.Saver(max_to_keep=10)
    if not tf.gfile.Exists(os.path.join(platform_cfg.CHECKPOINT_DIR, task_cfg.NAME)):
        tf.gfile.MakeDirs(os.path.join(platform_cfg.CHECKPOINT_DIR, task_cfg.NAME))

    with tf.Session(config=platform_cfg.get_sess_cfg()) as sess:
        if not tf.gfile.Exists(os.path.join(platform_cfg.LOG_DIR, task_cfg.NAME)):
            tf.gfile.MakeDirs(os.path.join(platform_cfg.LOG_DIR, task_cfg.NAME))
        summary_writer = tf.summary.FileWriter(os.path.join(platform_cfg.LOG_DIR, task_cfg.NAME))
        sess.run(tf.global_variables_initializer())
        sess.run(dataset_init)

        # lft = 10
        pbar = tqdm(range(task_cfg.NUM_EPOCHS))
        for epoch in pbar:
            total_loss_arr = []
            epoch_pbar = tqdm(range(dataset_cfg.NUM_TRAIN_EXAMPLES // task_cfg.BATCH_SIZE))
            for epoch_p in epoch_pbar:
                # lft -= 1
                # if lft < 0:
                #     break
                # print('step %d start' % (epoch_p,))
                _, _, _, total_loss_value, _ = sess.run([heatmap_loss, size_loss, offset_loss, total_loss, train_op])
                # _, _, _, total_loss_value = sess.run([heatmap_loss, size_loss, offset_loss, total_loss])
                epoch_pbar.set_description_str("Loss: %.3f" % total_loss_value)
                total_loss_arr.append(total_loss_value)
                # print('step %d end' % (epoch_p,))
                if (epoch_p + epoch * (dataset_cfg.NUM_TRAIN_EXAMPLES // task_cfg.BATCH_SIZE)) % 10000 == 9999:
                    saver.save(
                        sess,
                        os.path.join(platform_cfg.CHECKPOINT_DIR, task_cfg.NAME,
                                     'CenterNet_loss%.3f' % np.mean(total_loss_arr)))
            pbar.set_description_str("Epoch %d, Loss %.3f" % (epoch + 1, np.mean(total_loss_arr)))
            # if lft < 0:
            #     break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test centernet')
    parser.add_argument('--platform', default='config.platform.cfg_gpu_2p')
    parser.add_argument('--dataset', default='config.dataset.cfg_voc_object_detection')
    parser.add_argument('--task', default='config.task.cfg_centernet_voc_object_detection')
    args = parser.parse_args()
    platform_cfg = importlib.import_module(args.platform)
    dataset_cfg = importlib.import_module(args.dataset)
    task_cfg = importlib.import_module(args.task)
    main(platform_cfg, dataset_cfg, task_cfg)
