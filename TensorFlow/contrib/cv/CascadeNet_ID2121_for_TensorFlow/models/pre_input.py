"""
pre input
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
import tensorflow as tf
import numpy as np
import random
import os
import h5py as h5
# import cv2
from utils import compressed_sensing as cs

# import matplotlib.pyplot as plt
# from random import choice

FLAGS = tf.flags.FLAGS


# flags=tf.app.flags
# FLAGS=flags.FLAGS
#
# flags.DEFINE_integer('image_size',378,'Image sample size in pixels')
# flags.DEFINE_integer('batch_size',2,'Number of samples per batch')
# flags.DEFINE_float('R_factor',4.0,'desired reduction/undersampling factor')
# flags.DEFINE_bool('data_augment',True,'data augment')

def complex_twochannel(im):
    """
    convert complex image to a two channel double array
    input:
    im: complex image
    return:two channel array 
    """
    image_shape = np.shape(im)
    im_real = np.reshape(
        np.real(im), [image_shape[0], image_shape[1], image_shape[2], 1])
    im_imag = np.reshape(
        np.imag(im), [image_shape[0], image_shape[1], image_shape[2], 1])
    img = np.concatenate([im_real, im_imag], axis=-1)
    return img


def r2c(inp):
    """  
    input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    if inp.dtype == 'float32':
        dtype = np.complex64
    else:
        dtype = np.complex128
    out = np.zeros(inp.shape[0:2], dtype=dtype)
    out = inp[..., 0] + 1j * inp[..., 1]
    return out


def div0(a, b):
    """ 
    This function handles division by zero 
    """
    c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return c


def normalize01(img):
    """
    Normalize the image between o and 1
    """
    if len(img.shape) == 3:
        nimg = len(img)
    else:
        nimg = 1
        r, c = img.shape
        img = np.reshape(img, (nimg, r, c))
    img2 = np.empty(img.shape, dtype=img.dtype)
    for i in range(nimg):
        img2[i] = div0(img[i] - img[i].min(), img[i].ptp())
        # img2[i]=(img[i]-img[i].min())/(img[i].max()-img[i].min())
    return np.squeeze(img2).astype(img.dtype)


def get_filenames(dir_file='', shuffle_filename=False, mode='.bmp'):
    """
    get image files name in the dir
    input: 
    dir_file: files path
    shuffle_filename: bool
    mode: file type
    return:
    list of file names
    """
    try:
        filenames = tf.gfile.ListDirectory(dir_file)
    except IOError:
        print('cannot get files from {0}'.format(dir_file))
        return []
    filenames = sorted(filenames)
    if shuffle_filename:
        random.shuffle(filenames)
    else:
        filenames = sorted(filenames)
    filenames = [os.path.join(dir_file, f) for f in filenames if f.endswith(mode)]
    return filenames


def gaussian_noise_layer(input_image, std):
    """
    add gaussian noise
    """
    noise = tf.random_normal(shape=tf.shape(
        input_image), mean=0.0, stddev=std, dtype=tf.float32)
    noise_image = input_image + noise
    noise_image = tf.clip_by_value(noise_image, 0, 255.0)
    return noise_image


def GenerateUndersampled(im, mask):
    """
    apply undersampling
    """
    scile = im.shape[0]
    feature = []
    for i in range(scile):
        im_k = np.fft.fft2(im[i])
        mask_c = np.complex64(mask[i] / np.max(mask[i]))
        feature_k = im_k * mask_c
        feature.append(np.fft.ifft2(feature_k))
    feature = np.array(feature)
    return feature


def get_right_images(path):
    """
    prepare dataset
    """
    # print('Reading data, Please wait...')
    filename = path
    with h5.File(filename) as f:
        label, mask = f['label'][:], f['mask'][:]
    feature = GenerateUndersampled(label, mask)
    feature = complex_twochannel(feature)
    label = complex_twochannel(label)

    return feature, label, mask


def data_augment(filenames):
    """
    data_augment
    """
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
    load_img = tf.keras.preprocessing.image.load_img
    img_to_array = tf.keras.preprocessing.image.img_to_array

    datagen = ImageDataGenerator(
        rotation_range=360,
        horizontal_flip=True,
        vertical_flip=True
    )
    for name in filenames:
        # this is a PIL image, please replace to your own file path
        img = load_img(name)
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        # this is a Numpy array with shape (1, 3, 150, 150)
        x = x.reshape((1,) + x.shape)
        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in datagen.flow(x,
                                  batch_size=1,
                                  save_to_dir='data/data_keen_augment',
                                  save_prefix='',
                                  save_format='bmp'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely


def nMse(org, recon):
    """
    this function calculates rmse between the original and the reconstruted images
    """
    diff = np.sum(np.square(np.abs(org - recon)))
    nMse_tmp = np.sqrt((diff / (np.sum(np.square(org)))))
    return nMse_tmp * nMse_tmp


def myPSNR(org, recon):
    """ 
    This function calculates PSNR between the original and
    the reconstructed images
    """
    mse = np.sum(np.square(np.abs(org - recon))) / org.size
    # psnr=20*np.log10(org.max()/(np.sqrt(mse)+1e-10 ))
    psnr = 20 * np.log10((org.max() - org.min()) / (np.sqrt(mse) + 1e-10))
    return psnr


def main():
    """
    tmp main
    """
    filenames_1 = get_filenames('ChestTrain', mode='.bmp')
    data_augment(filenames_1)


if __name__ == '__main__':
    tf.app.run()
