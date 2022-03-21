# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from scipy.ndimage import gaussian_filter

FLAGS = tf.flags.FLAGS
DEFAULT_DATALOADER = 'basic_loader'
DEFAULT_MODEL = 'bsrn'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    tf.flags.DEFINE_string('dataloader', DEFAULT_DATALOADER, 'Name of the data loader.')
    tf.flags.DEFINE_string('model', DEFAULT_MODEL, 'Name of the model.')
    tf.flags.DEFINE_string('scales', '2,3,4',
                           'Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')
    tf.flags.DEFINE_string("chip", "gpu", "Run on which chip, (npu or gpu or cpu)")
    tf.flags.DEFINE_string("output_path", "output", "output image data path")
    tf.flags.DEFINE_string("truth_path", "truth image bin", "output image data path")
    tf.flags.DEFINE_string('save_path', None,
                           'Base path of the upscaled images. Specify this to save the upscaled images.')
    tf.flags.DEFINE_integer('shave_size', 4,
                            'Amount of pixels to crop the borders of the images before calculating quality metrics.')
    tf.flags.DEFINE_boolean('ensemble_only', False, 'Calculate (and save) ensembled image only.')
    # parse data loader and model first and import them
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--dataloader', default=DEFAULT_DATALOADER)
    pre_parser.add_argument('--model', default=DEFAULT_MODEL)
    pre_parsed = pre_parser.parse_known_args()[0]
    if (pre_parsed.dataloader is not None):
        DATALOADER_MODULE = importlib.import_module('dataloaders.' + pre_parsed.dataloader)
    if (pre_parsed.model is not None):
        MODEL_MODULE = importlib.import_module('models.' + pre_parsed.model)

# image saving session
tf_image_save_graph = tf.Graph()
with tf_image_save_graph.as_default():
    tf_image_save_path = tf.placeholder(tf.string, [])
    tf_image_save_image = tf.placeholder(tf.float32, [None, None, 3])

    tf_image = tf_image_save_image
    tf_image = tf.round(tf_image)
    tf_image = tf.clip_by_value(tf_image, 0, 255)
    tf_image = tf.cast(tf_image, tf.uint8)

    tf_image_png = tf.image.encode_png(tf_image)
    tf_image_save_op = tf.write_file(tf_image_save_path, tf_image_png)

    tf_image_save_init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    tf_image_save_session = tf.Session(config=config)
    tf_image_save_session.run(tf_image_save_init)


def _clip_image(image):
    return np.clip(np.round(image), a_min=0, a_max=255)


def _shave_image(image, shave_size=4):
    return image[shave_size:-shave_size, shave_size:-shave_size]


def _fit_truth_image_size(output_image, truth_image):
    return truth_image[0:output_image.shape[0], 0:output_image.shape[1]]


def _image_psnr(output_image, truth_image):
    diff = truth_image - output_image
    mse = np.mean(np.power(diff, 2))
    psnr = 10.0 * np.log10(255.0 ** 2 / mse)
    return psnr

def _image_psnr2(output_image, truth_image):
    yr_out = 0.257*output_image[:,:,0] + 0.504*output_image[:,:,1] + 0.098*output_image[:,:,2] + 16.5
    yr_tr = 0.257*truth_image[:,:,0] + 0.504*truth_image[:,:,1] + 0.098*truth_image[:,:,2] + 16.5
    diff = yr_tr - yr_out
    mse = np.mean(np.power(diff, 2))
    psnr = 10.0 * np.log10(255.0 ** 2 / mse)
    return psnr

def _image_rmse(output_image, truth_image):
    diff = truth_image - output_image
    rmse = np.sqrt(np.mean(diff ** 2))
    return rmse

def _image_ssim(X, Y):
    """
       Computes the mean structural similarity between two images.
    """
    assert (X.shape == Y.shape), "Image-patche provided have different dimensions"
    nch = 1 if X.ndim == 2 else X.shape[-1]
    mssim = []
    for ch in range(nch):
        Xc, Yc = X[..., ch].astype(np.float64), Y[..., ch].astype(np.float64)
        mssim.append(compute_ssim(Xc, Yc))
    return np.mean(mssim)


def compute_ssim(X, Y):
    """
       Compute the structural similarity per single channel (given two images)
    """
    # variables are initialized as suggested in the paper
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 5

    # means
    ux = gaussian_filter(X, sigma)
    uy = gaussian_filter(Y, sigma)

    # variances and covariances
    uxx = gaussian_filter(X * X, sigma)
    uyy = gaussian_filter(Y * Y, sigma)
    uxy = gaussian_filter(X * Y, sigma)

    # normalize by unbiased estimate of std dev
    N = win_size ** X.ndim
    unbiased_norm = N / (N - 1)  # eq. 4 of the paper
    vx = (uxx - ux * ux) * unbiased_norm
    vy = (uyy - uy * uy) * unbiased_norm
    vxy = (uxy - ux * uy) * unbiased_norm

    R = 255
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    # compute SSIM (eq. 13 of the paper)
    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim / D
    mssim = SSIM.mean()

    return mssim
# data loader
dataloader = DATALOADER_MODULE.create_loader()
dataloader.prepare()
num_images = dataloader.get_num_images()
scale_list = list(map(lambda x: int(x), FLAGS.scales.split(',')))
scale = scale_list[0]
num_total_outputs = FLAGS.bsrn_recursions // FLAGS.bsrn_recursion_frequency
print("num_total_outputs:",num_total_outputs)

modules_average_psnr_dict = {}
modules_average_ssim_dict = {}

for scale in scale_list:
    modules_average_psnr_dict[scale] = []
    modules_average_ssim_dict[scale] = []

psnr_list = []
ssim_list = []

for i in range(num_total_outputs + 1):
    psnr_list.append([])
    ssim_list.append([])

for i, image_name in enumerate(dataloader.image_name_list):
    print(i, image_name)
    output_name = os.path.join(FLAGS.output_path, image_name + '.data_output_0.bin')
    tmp = np.fromfile(output_name, dtype=np.float32)
    new_tmp = tmp.reshape(16, 480, 320, 3)
    output_images = np.array_split(new_tmp, 16, axis=0)
    print(type(output_images))
    print(len(output_images))
    # _, truth_image, truth_image_name = dataloader.get_image_pair(i, scale_list[0])
    truth_name = os.path.join(FLAGS.truth_path,image_name+'.data.bin')
    tmp = np.fromfile(truth_name, dtype=np.float32)
    truth_image = tmp.reshape(480, 320, 3)

    print("truth_image_shape:",truth_image.shape)

    output_image_ensemble = np.zeros_like(output_images[0][0])
    ensemble_factor_total = 0.0
    # testing on single set of image_outputs
    for j in range(num_total_outputs):
        num_recursions = (j + 1) * FLAGS.bsrn_recursion_frequency
        output_image = output_images[j][0]

        ensemble_factor = 1.0 / (2.0 ** (num_total_outputs - num_recursions))
        output_image_ensemble = output_image_ensemble + (output_image * ensemble_factor)
        ensemble_factor_total += ensemble_factor

        if not FLAGS.ensemble_only:
            if FLAGS.save_path is not None:
                output_image_path = os.path.join(FLAGS.save_path, 't%d' % num_recursions, 'x%d' % scale,
                                                 os.path.splitext(image_name)[0] + '.png')
                tf_image_save_session.run(tf_image_save_op, feed_dict={tf_image_save_path: output_image_path,
                                                                       tf_image_save_image: output_image})

        truth_image = _clip_image(truth_image)
        output_image = _clip_image(output_image)

        truth_image = _fit_truth_image_size(output_image=output_image, truth_image=truth_image)

        truth_image_shaved = _shave_image(truth_image, shave_size=FLAGS.shave_size)
        output_image_shaved = _shave_image(output_image, shave_size=FLAGS.shave_size)

        psnr = _image_psnr2(output_image=output_image_shaved, truth_image=truth_image_shaved)
        # ssim = _image_rmse(im1=output_image_shaved, im2=truth_image_shaved)
        ssim = _image_ssim(output_image_shaved, truth_image_shaved)

        tf.logging.info('t%d, x%d, %d/%d, psnr=%.2f, ssim=%.2f' % (
            num_recursions, scale, i + 1, num_images, psnr, ssim))
        print('t%d, x%d, %d/%d, psnr=%.2f, ssim=%.2f' % (num_recursions, scale, i + 1, num_images, psnr, ssim))
        psnr_list[j].append(psnr)
        ssim_list[j].append(ssim)

    output_image = output_image_ensemble / ensemble_factor_total

    if (FLAGS.save_path is not None):
        output_image_path = os.path.join(FLAGS.save_path, 'ensemble', 'x%d' % scale, os.path.splitext(image_name)[0] + '.png')
        tf_image_save_session.run(tf_image_save_op, feed_dict={tf_image_save_path:output_image_path, tf_image_save_image:output_image})

    truth_image = _clip_image(truth_image)
    output_image = _clip_image(output_image)

    truth_image = _fit_truth_image_size(output_image=output_image, truth_image=truth_image)

    truth_image_shaved = _shave_image(truth_image, shave_size=FLAGS.shave_size)
    output_image_shaved = _shave_image(output_image, shave_size=FLAGS.shave_size)

    psnr = _image_psnr2(output_image=output_image_shaved, truth_image=truth_image_shaved)
    # ssim = _image_rmse(im1=output_image_shaved, im2=truth_image_shaved)
    ssim = _image_ssim(output_image_shaved, truth_image_shaved)
    tf.logging.info('ensemble, x%d, %d/%d, psnr=%.2f, ssim=%.2f' % (scale, i+1, num_images, psnr, ssim))
    print('ensemble, x%d, %d/%d, psnr=%.2f, ssim=%.2f' % (scale, i+1, num_images, psnr, ssim))
    psnr_list[num_total_outputs].append(psnr)
    ssim_list[num_total_outputs].append(ssim)

for i in range(num_total_outputs+1):
    average_psnr = np.mean(psnr_list[i])
    modules_average_psnr_dict[scale].append(average_psnr)
    average_ssim = np.mean(ssim_list[i])
    modules_average_ssim_dict[scale].append(average_ssim)
tf.logging.info('finished')
print('finished')
for scale in scale_list:
    print('- x%d, PSNR and SSIM:' % (scale))
    print(','.join([('%.3f' % x) for x in modules_average_psnr_dict[scale]]))
    print('')
    print(','.join([('%.3f' % x) for x in modules_average_ssim_dict[scale]]))