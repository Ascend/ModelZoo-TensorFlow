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

import skimage
from skimage.transform import resize
from PIL import Image
import time
import tensorflow as tf
from zipfile import ZipFile
from tensorflow.python.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from npu_bridge.npu_init import *

global graph, sess


def extract_zip(input_zip):
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}


def DepthNorm(x, maxDepth):
    return maxDepth / x


def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2, is_distributed=False):
    # Support multiple RGBs, one RGB image, even grayscale 
    if len(images.shape) < 3: images = np.stack((images, images, images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    global graph, sess
    graph = tf.get_default_graph()
    sess = K.get_session()
    with sess.as_default():
        with graph.as_default():
            # start = time.time()
            callbacks = []
            if is_distributed:
                callbacks.append(NPUBroadcastGlobalVariablesCallback(0))
            predictions = model.predict(images, batch_size=batch_size, callbacks=callbacks)
            # time_run = time.time() - start
    # Put in expected range
    # return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth, time_run
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth


def scale_up(scale, images):
    scaled = []

    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))

    return np.stack(scaled)


def image_resize(img, resolution=480, padding=6):
    return resize(img, (resolution, int(resolution * 4 / 3)), preserve_range=True, mode='reflect',
                  anti_aliasing=True)


def load_images(image_files, resolution=480):
    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open(file), dtype=float) / 255, 0, 1)
        x = image_resize(x, resolution=resolution)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)


def to_multichannel(i):
    if i.shape[2] == 3:
        return i
    i = i[:, :, 0]
    return np.stack((i, i, i), axis=2)


def display_images(outputs, inputs=None, gt=None, is_colormap=True, is_rescale=True):
    plasma = plt.get_cmap('plasma')

    shape = (outputs[0].shape[0], outputs[0].shape[1], 3)

    all_images = []

    for i in range(outputs.shape[0]):
        imgs = []

        if isinstance(inputs, (list, tuple, np.ndarray)):
            x = to_multichannel(inputs[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True)
            imgs.append(x)

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = to_multichannel(gt[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True)
            imgs.append(x)

        if is_colormap:
            rescaled = outputs[i][:, :, 0]
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            imgs.append(plasma(rescaled)[:, :, :3])
        else:
            imgs.append(to_multichannel(outputs[i]))

        img_set = np.hstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)

    return skimage.util.montage(all_images, multichannel=True, fill=(0, 0, 0))


def save_images(filename, outputs, inputs=None, gt=None, is_colormap=True, is_rescale=False):
    montage = display_images(outputs, inputs, is_colormap, is_rescale)
    im = Image.fromarray(np.uint8(montage * 255))
    im.save(filename)


def load_test_data(test_data_zip_file='nyu_test.zip'):
    data = extract_zip(test_data_zip_file)
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
    depth = np.load(BytesIO(data['eigen_test_depth.npy']))
    crop = np.load(BytesIO(data['eigen_test_crop.npy']))
    return {'rgb': rgb, 'depth': depth, 'crop': crop}


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10


def evaluate(model, rgb, depth, crop, batch_size=6, verbose=False, is_distributed=False):
    N = len(rgb)

    bs = batch_size

    predictions = []
    testSetDepths = []

    # test_time = 0
    # test_n = 0

    for i in range(N // bs):
        x = rgb[i * bs:(i + 1) * bs, :, :, :]

        # Compute results
        true_y = depth[i * bs:(i + 1) * bs, :, :]
        pred_y = scale_up(2, predict(model, x / 255, minDepth=10, maxDepth=1000, batch_size=bs, is_distributed=is_distributed)[:, :, :, 0]) * 10.0

        # pred_y, test_time_y = predict(model, x / 255, minDepth=10, maxDepth=1000, batch_size=bs)
        # pred_y = scale_up(2, pred_y[:, :, :, 0]) * 10.0
        # print('test_time_y: {}'.format(test_time_y))

        # Test time augmentation: mirror image estimate
        pred_y_flip = scale_up(2, predict(model, x[..., ::-1, :] / 255, minDepth=10, maxDepth=1000,
                                          batch_size=bs, is_distributed=is_distributed)[:, :, :, 0]) * 10.0

        # pred_y_flip, test_time_y_flip = predict(model, x[..., ::-1, :] / 255, minDepth=10, maxDepth=1000, batch_size=bs)
        # pred_y_flip = scale_up(2, pred_y_flip[:, :, :, 0]) * 10.0
        # print('test_time_y_flip: {}'.format(test_time_y_flip))

        # if i >= 5:
        #     test_time += (test_time_y + test_time_y_flip)
        #     test_n += 1

        # Crop based on Eigen et al. crop
        true_y = true_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
        pred_y = pred_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
        pred_y_flip = pred_y_flip[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]

        # Compute errors per image in batch
        for j in range(len(true_y)):
            predictions.append((0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j])))
            testSetDepths.append(true_y[j])

    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)

    e = compute_errors(predictions, testSetDepths)

    # print('[info] validate average time images/sec: {}'.format((test_n * 2 * bs) / test_time))

    if verbose:
        print(" a1 %-10.4f a2 %-10.4f a3 %-10.4f rel %-10.4f rms %-10.4f log_10 %-10.4f\n" % (e[0], e[1], e[2], e[3],
                                                                                              e[4], e[5]))
    return e
