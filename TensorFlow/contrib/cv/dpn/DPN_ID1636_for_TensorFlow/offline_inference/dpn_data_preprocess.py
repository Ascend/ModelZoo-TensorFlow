# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import numpy as np
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

is_training = False
batch_size = 9
epochs = 1
image_num = 1449
# _HEIGHT = 512
# _WIDTH = 512
# img_N = 1449


def read_data(tf_file, is_training):
    def _parse_read(tfrecord_file):
        features = {
            'image':
                tf.io.FixedLenFeature((), tf.string),
            "label":
                tf.io.FixedLenFeature((), tf.string),
            "mask":
                tf.io.FixedLenFeature((), tf.string),
            'height':
                tf.io.FixedLenFeature((), tf.int64),
            'width':
                tf.io.FixedLenFeature((), tf.int64),
            'channels':
                tf.io.FixedLenFeature((), tf.int64)
        }
        parsed = tf.io.parse_single_example(tfrecord_file, features)
        image = tf.decode_raw(parsed['image'], tf.uint8)
        image = tf.reshape(image, [parsed['height'], parsed['width'], parsed['channels']])
        label = tf.decode_raw(parsed['label'], tf.uint8)
        label = tf.reshape(label, [parsed['height'], parsed['width'], parsed['channels']])
        mask = tf.decode_raw(parsed['mask'], tf.uint8)
        mask = tf.reshape(mask, [parsed['height'], parsed['width'], 24])
        label = label[:, :, 0:1]
        mask = mask[:, :, 0:1]

        image, label, mask = _augmentation(image, label, mask, parsed['height'], parsed['width'])
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int64)
        mask = tf.cast(mask, tf.float32)
        image, label, mask = _preprocess(image, label, mask)
        return image, label[:, :, 0], mask[:, :, 0]

    def _augmentation(image, label, mask, h, w):
        image = tf.image.resize_images(image, size=[512, 512], method=0)
        image = tf.cast(image, tf.uint8)
        label = tf.image.resize_images(label, size=[64, 64], method=1)
        mask = tf.image.resize_images(mask, size=[64, 64], method=1)
        # 随机翻转
        with tf.Session() as sess:
            rand_value = tf.random.uniform(()).eval()
     
        if rand_value > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
            mask = tf.image.flip_left_right(mask)
        return image, label, mask

    def _preprocess(image, label, mask):
        image = image - [122.67891434, 116.66876762, 104.00698793]
        image = image / 255.
        return image, label, mask

    dataset = tf.data.TFRecordDataset(tf_file, num_parallel_reads=4)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size * 10))
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(map_func=_parse_read, batch_size=batch_size, drop_remainder=True,
                                      num_parallel_calls=2))
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch, masks_batch = iterator.get_next()
    return images_batch, labels_batch


if __name__ == '__main__':

    data_path = sys.argv[1]
    output_path = sys.argv[2]
    output_path += "/"


    clear = True
    if clear:
        os.system("rm -rf "+output_path+"data")
        os.system("rm -rf "+output_path+"label")
    if os.path.isdir(output_path+"data"):
        pass
    else:
        os.makedirs(output_path+"data")
    if os.path.isdir(output_path+"label"):
        pass
    else:
        os.makedirs(output_path+"label")
    images_batch, labels_batch = read_data(data_path,is_training)
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    image = []
    label = []
    for step in tqdm(range(int(image_num / batch_size))):
        x_in, y_in = sess.run([images_batch, labels_batch])
        label.append(y_in)
        x_in.tofile(output_path+"data/"+str(step)+".bin")
    label = np.array(label)
    np.save(output_path + "label/label.npy", label)
    print("[info]  data bin ok")