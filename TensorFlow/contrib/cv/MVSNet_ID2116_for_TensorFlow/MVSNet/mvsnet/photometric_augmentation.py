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
#
from npu_bridge.npu_init import *
import cv2 as cv
import numpy as np
import tensorflow as tf

augmentations = [
        # 'additive_gaussian_noise',
        # 'additive_speckle_noise',
        'random_brightness',
        'random_contrast',
        # 'additive_shade',
        'motion_blur'
]

def additive_gaussian_noise(image, stddev_range=[5, 95]):
    stddev = tf.random_uniform((), *stddev_range)
    noise = tf.random_normal(tf.shape(image), stddev=stddev)
    noisy_image = tf.clip_by_value(image + noise, 0, 255)
    return noisy_image


def additive_speckle_noise(image, prob_range=[0.0, 0.005]):
    prob = tf.random_uniform((), *prob_range)
    sample = tf.random_uniform(tf.shape(image))
    noisy_image = tf.where(sample <= prob, tf.zeros_like(image), image)
    noisy_image = tf.where(sample >= (1. - prob), 255.*tf.ones_like(image), noisy_image)
    return noisy_image


def random_brightness(image, max_abs_change=50):
    return tf.clip_by_value(tf.image.random_brightness(image, max_abs_change), 0, 255)


def random_contrast(image, strength_range=[0.5, 1.5]):
    return tf.clip_by_value(tf.image.random_contrast(image, *strength_range), 0, 255)


def additive_shade(image, nb_ellipses=20, transparency_range=[-0.5, 0.8],
                   kernel_size_range=[250, 350]):

    def _py_additive_shade(img):
        min_dim = min(img.shape[:2]) / 4
        mask = np.zeros(img.shape[:2], np.uint8)
        for i in range(nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
            y = np.random.randint(max_rad, img.shape[0] - max_rad)
            angle = np.random.rand() * 90
            cv.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

        transparency = np.random.uniform(*transparency_range)
        kernel_size = np.random.randint(*kernel_size_range)
        if (kernel_size % 2) == 0:  # kernel_size has to be odd
            kernel_size += 1
        mask = cv.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        shaded = img * (1 - transparency * mask[..., np.newaxis]/255.)
        return np.clip(shaded, 0, 255)

    shaded = tf.py_func(_py_additive_shade, [image], tf.float32)
    res = tf.reshape(shaded, tf.shape(image))
    return res


def motion_blur(image, max_kernel_size=10):

    def _py_motion_blur(img):
        # Either vertial, hozirontal or diagonal blur
        mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
        ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
        center = int((ksize-1)/2)
        kernel = np.zeros((ksize, ksize))
        if mode == 'h':
            kernel[center, :] = 1.
        elif mode == 'v':
            kernel[:, center] = 1.
        elif mode == 'diag_down':
            kernel = np.eye(ksize)
        elif mode == 'diag_up':
            kernel = np.flip(np.eye(ksize), 0)
        var = ksize * ksize / 16.
        grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
        gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
        kernel *= gaussian
        kernel /= np.sum(kernel)
        img = cv.filter2D(img, -1, kernel)
        return img

    blurred = tf.numpy_function(_py_motion_blur, [image], tf.float32)
    return tf.reshape(blurred, tf.shape(image))

def online_augmentation(image, random_order=True):
    primitives = augmentations
    config = {}
    config['random_brightness'] = {'max_abs_change': 50}
    config['random_contrast'] = {'strength_range': [0.3, 1.5]}
    config['additive_gaussian_noise'] = {'stddev_range': [0, 10]}
    config['additive_speckle_noise'] = {'prob_range': [0, 0.0035]}
    config['additive_shade'] = {'transparency_range': [-0.5, 0.5], 'kernel_size_range': [100, 150]}
    config['motion_blur'] = {'max_kernel_size': 3}

    with tf.name_scope('online_augmentation'):
        prim_configs = [config.get(p, {}) for p in primitives]

        indices = tf.range(len(primitives))
        if random_order:
            indices = tf.random.shuffle(indices)

        def step(i, image):
            fn_pairs = [(tf.equal(indices[i], j), lambda p=p, c=c: getattr(photaug, p)(image, **c))
                        for j, (p, c) in enumerate(zip(primitives, prim_configs))]
            image = tf.case(fn_pairs)
            return i + 1, image

        _, aug_image = tf.while_loop(lambda i, image: tf.less(i, len(primitives)),
                                     step, [0, image], parallel_iterations=1)

    return aug_image
