import os
import numpy as np
from tqdm import tqdm
from absl import flags
import tensorflow as tf
from os.path import join
from .config import get_config
from npu_bridge.npu_init import *
from .datasets.common import read_images_from_tfrecords


flags.DEFINE_string('test_data_dir', None, 'where to load test data')
flags.DEFINE_string('save_dir', None, 'where to save converted data')


def main(config, sess):
    print('Start converting ...')
    save_image_dir = join(config.save_dir, 'input')
    os.makedirs(save_image_dir, exist_ok=True)
    all_gt3ds = []
    counter = 0
    all_tfrecords = sorted(
        [join(config.test_data_dir, path) for path in os.listdir(config.test_data_dir)]
        )
    for n, record in enumerate(all_tfrecords):
        images, gt3ds = get_data(record, sess)
        all_gt3ds.append(gt3ds)
        for i in tqdm(range(len(images)), desc='Record %d' % n):
            image = images[i:i+1, :, :, :].astype(np.float32)
            image.tofile(
                join(save_image_dir, '%04d.bin' % counter)
                )
            counter += 1
    all_gt3ds = np.vstack(all_gt3ds)
    assert all_gt3ds.shape[0] == counter
    np.save(
        join(config.save_dir, 'test_label.npy'),
        all_gt3ds
        )
    print('Done, total images=%d' % counter)

def get_data(record, sess):
    images, kps, gt3ds = read_images_from_tfrecords(record, img_size=config.img_size, sess=sess)
    return images, gt3ds


if __name__ == '__main__':
    config = get_config()
    # define sess config
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # close remap
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    sess = tf.Session(config=sess_config)
    main(config, sess)
    sess.close()