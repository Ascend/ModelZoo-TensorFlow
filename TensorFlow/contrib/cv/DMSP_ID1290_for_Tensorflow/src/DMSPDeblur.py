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
import time
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import scipy.signal as sig
import tensorflow as tf


def computePSNR(img1, img2, pad_y, pad_x):
    """ Computes peak signal-to-noise ratio between two images. 
    Input:
    img1: First image in range of [0, 255].
    img2: Second image in range of [0, 255].
    pad_y: Scalar radius to exclude boundaries from contributing to PSNR computation in vertical direction.
    pad_x: Scalar radius to exclude boundaries from contributing to PSNR computation in horizontal direction.
    
    Output: PSNR """

    img1_u = (np.clip(np.squeeze(img1), 0, 255.0)[pad_y:-pad_y,pad_x:-pad_x,:]).astype(dtype=np.uint8)
    img2_u = (np.clip(np.squeeze(img2), 0, 255.0)[pad_y:-pad_y,pad_x:-pad_x,:]).astype(dtype=np.uint8)
    imdiff = (img1_u).astype(dtype=np.float32) - (img2_u).astype(dtype=np.float32)
    rmse = np.sqrt(np.mean(np.power(imdiff[:], 2)))
    return 20.0 * np.log10(255.0 / rmse)

def filter_image(image, kernel, mode='valid'):
    """ Implements color filtering (convolution using a flipped kernel) """
    chs = []
    for d in range(image.shape[2]):
        channel = sig.convolve2d(image[:,:,d], np.flipud(np.fliplr(kernel)), mode=mode)
        chs.append(channel)
    return np.stack(chs, axis=2)

def convolve_image(image, kernel, mode='valid'):
    """ Implements color image convolution """
    chs = []
    for d in range(image.shape[2]):
        channel = sig.convolve2d(image[:,:,d], kernel, mode=mode)
        chs.append(channel)
    return np.stack(chs, axis=2)


def DMSPDeblur(degraded, kernel, sigma_d, params):
    """ Implements stochastic gradient descent (SGD) Bayes risk minimization for image deblurring described in:
     "Deep Mean-Shift Priors for Image Restoration" (http://home.inf.unibe.ch/~bigdeli/DMSPrior.html)
     S. A. Bigdeli, M. Jin, P. Favaro, M. Zwicker, Advances in Neural Information Processing Systems (NIPS), 2017

     Input:
     degraded: Observed degraded RGB input image in range of [0, 255].
     kernel: Blur kernel (internally flipped for convolution).
     sigma_d: Noise standard deviation. (set to -1 for noise-blind deblurring)
     params: Set of parameters.
     params.denoiser: The denoiser function hanlde.

     Optional parameters:
     params.sigma_dae: The standard deviation of the denoiser training noise. default: 11
     params.num_iter: Specifies number of iterations.
     params.mu: The momentum for SGD optimization. default: 0.9
     params.alpha the step length in SGD optimization. default: 0.1

     Outputs:
     res: Solution."""

    if 'denoiser' not in params:
        raise ValueError('Need a denoiser in params.denoiser!')

    if 'gt' in params:
        print_iter = True
    else:
        print_iter = False

    if 'sigma_dae' not in params:
        params['sigma_dae'] = 11.0

    if 'num_iter' not in params:
        params['num_iter'] = 10

    if 'mu' not in params:
        params['mu'] = 0.9

    if 'alpha' not in params:
        params['alpha'] = 0.1

    pad_y = np.floor(kernel.shape[0] / 2.0).astype(np.int64)
    pad_x = np.floor(kernel.shape[1] / 2.0).astype(np.int64)
    res = np.pad(degraded, pad_width=((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='edge').astype(np.float32)

    step = np.zeros(res.shape)

    if print_iter:
        psnr = computePSNR(params['gt'], res, pad_y, pad_x)
        print('Initialized with PSNR: ' + str(psnr))

    for iter in range(params['num_iter']):
        if print_iter:
            # print('Running iteration: ' + str(iter))
            t = time.time()

        #     compute prior gradient
        noise = np.random.normal(0.0, params['sigma_dae'], res.shape).astype(np.float32)

        #离线推理
        input_image = res + noise
        input_image.tofile("/cache/model/dmsp_input_image.bin")  # 处理后的图片保存为bin文件

        rec = params['denoiser'].denoise(res + noise,False)
        prior_grad = res - rec

        #     compute data gradient
        map_conv = filter_image(res, kernel)
        data_err = map_conv - degraded
        data_grad = convolve_image(data_err, kernel, mode='full')

        relative_weight = 0.5
        if sigma_d < 0:
            sigma2 = 2 * params['sigma_dae'] * params['sigma_dae']

            data_sum = np.sum(np.power(np.int64(data_err[:]), 2))
            kernel_sum = np.sum(np.power(np.int64(kernel[:]), 2))
            lambda_ = (degraded.size) / (data_sum
                                         + degraded.size * sigma2 * kernel_sum)
            relative_weight = lambda_ / (lambda_ + 1 / params['sigma_dae'] / params['sigma_dae'])
        else:
            relative_weight = (1 / sigma_d / sigma_d) / (
                    1 / sigma_d / sigma_d + 1 / params['sigma_dae'] / params['sigma_dae'])
        #     sum the gradients
        grad_joint = data_grad * relative_weight + prior_grad * (1 - relative_weight)

        #     update
        step = params['mu'] * step - params['alpha'] * grad_joint
        res = res + step
        res = np.minimum(255.0, np.maximum(0, res)).astype(np.float32)

        psnr = computePSNR(params['gt'], res, pad_y, pad_x)
    if print_iter:
        # print ('PSNR is: ' + str(psnr) + ', iteration finished in ' + str(time.time() - t) + ' seconds')
        print('Finished psnr = %.2f (%.1f examples/sec; %.3f sec/batch)' % (
        ( psnr, 1 / (time.time() - t), (time.time() - t))))


    return res,psnr
