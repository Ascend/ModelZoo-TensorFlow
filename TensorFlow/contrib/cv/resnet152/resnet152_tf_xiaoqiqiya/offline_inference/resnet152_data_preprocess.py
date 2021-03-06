# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import numpy as np
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

is_training = False
batch_size = 50
epochs = 1
image_num = 50000



def _parse_read(example_proto):
    features = {"image": tf.FixedLenFeature([], tf.string, default_value=""),
                "height": tf.FixedLenFeature([], tf.int64, default_value=[0]),
                "width": tf.FixedLenFeature([], tf.int64, default_value=[0]),
                "channels": tf.FixedLenFeature([], tf.int64, default_value=[3]),
                "colorspace": tf.FixedLenFeature([], tf.string, default_value=""),
                "img_format": tf.FixedLenFeature([], tf.string, default_value=""),
                "label": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "bbox_xmin": tf.VarLenFeature(tf.float32),
                "bbox_xmax": tf.VarLenFeature(tf.float32),
                "bbox_ymin": tf.VarLenFeature(tf.float32),
                "bbox_ymax": tf.VarLenFeature(tf.float32),
                "text": tf.FixedLenFeature([], tf.string, default_value=""),
                "filename": tf.FixedLenFeature([], tf.string, default_value="")
                }

    parsed_features = tf.parse_single_example(example_proto, features)
    label = parsed_features["label"]
    images = tf.image.decode_jpeg(parsed_features["image"])
    h = tf.cast(parsed_features['height'], tf.int64)
    w = tf.cast(parsed_features['width'], tf.int64)
    c = tf.cast(parsed_features['channels'], tf.int64)
    images = tf.reshape(images, [h, w, 3])
    images = tf.cast(images, tf.float32)
    images = tf.image.central_crop(images, central_fraction=0.85)
    images = tf.image.resize_images(images, [224, 224])
    return images, label



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

    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_read,num_parallel_calls=4)
    dataset = dataset.repeat(1)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    image = []
    label = []
    for step in range(int(image_num / batch_size)):
        x_in, y_in = sess.run([images_batch, labels_batch])
        y_in = np.squeeze(y_in, 1)
        x_in.tofile(output_path+"data/"+str(step)+".bin")
        label += y_in.tolist()
    label = np.array(label)
    np.save(output_path + "label/imageLabel.npy", label)
    print("[info]  data bin ok")