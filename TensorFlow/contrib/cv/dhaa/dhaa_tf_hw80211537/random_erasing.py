import time
import numpy as np
import tensorflow as tf

def random_erasing(img, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
    '''
    img is a 3-D variable (ex: tf.Variable(image, validate_shape=False) ) and  HWC order
    '''
    # HWC order
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    channel = tf.shape(img)[2]
    area = tf.cast(width*height, tf.float32)

    erase_area_low_bound = tf.cast(tf.round(tf.sqrt(sl * area * r1)), tf.int32)
    erase_area_up_bound = tf.cast(tf.round(tf.sqrt((sh * area) / r1)), tf.int32)
    h_upper_bound = tf.minimum(erase_area_up_bound, height)
    w_upper_bound = tf.minimum(erase_area_up_bound, width)

    h = tf.random.uniform([], erase_area_low_bound, h_upper_bound, tf.int32)
    w = tf.random.uniform([], erase_area_low_bound, w_upper_bound, tf.int32)

    x1 = tf.random.uniform([], 0, height+1 - h, tf.int32)
    y1 = tf.random.uniform([], 0, width+1 - w, tf.int32)

    erase_area = tf.cast(tf.random.uniform([h, w, channel], 0, 255, tf.int32), tf.uint8)

    erasing_img = img[x1:x1+h, y1:y1+w, :].assign(erase_area)

    return tf.cond(tf.random.uniform([], 0, 1) > probability, lambda: img, lambda: erasing_img)


def random_erase_np_v2(img, probability = 0.3, sl = 0.02, sh = 0.3, r1 = 0.3):

    if np.random.uniform() > probability:
        return img

    height = img.shape[0]
    width = img.shape[1]
    channel = img.shape[2]
    area = width * height

    erase_area_low_bound = np.round( np.sqrt(sl * area * r1) ).astype(np.int)
    erase_area_up_bound = np.round( np.sqrt((sh * area) / r1) ).astype(np.int)
    if erase_area_up_bound < height:
        h_upper_bound = erase_area_up_bound
    else:
        h_upper_bound = height
    if erase_area_up_bound < width:
        w_upper_bound = erase_area_up_bound
    else:
        w_upper_bound = width

    h = np.random.randint(erase_area_low_bound, h_upper_bound)
    w = np.random.randint(erase_area_low_bound, w_upper_bound)

    x1 = np.random.randint(0, height+1 - h)
    y1 = np.random.randint(0, width+1 - w)

    x1 = np.random.randint(0, height - h)
    y1 = np.random.randint(0, width - w)
    # img_ori = img * 1
    img[x1:x1+h, y1:y1+w, :] = np.random.randint(0, 255, size=(h, w, channel)).astype(np.uint8)

    return img
