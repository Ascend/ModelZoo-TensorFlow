# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import numpy as np
import cv2
import random
import time

from src.utils.degradation.base import Base


class NoiseAugmentation(Base):
    """Noise addition class.
    """
    def __init__(self, **kwargs):
        self.max_value = 255.
        self.min_value = 0.    # input data should be in range [min_value, max_value]
        self._data_format = 'hwc'   # ['hwc', 'thwc']

    def get_height(self, im):
        return im.shape[self._data_format.index['h']]

    def get_width(self, im):
        return im.shape[self._data_format.index['w']]

    def get_temp(self, im):
        return im.shape[self._data_format.index['t']]

    @property
    def data_format(self):
        return self._data_format

    @data_format.setter
    def data_format(self, target_format):
        self._data_format = target_format.lower()

    def __call__(self, im, **kwargs):
        # Input im should be in range [0, 255], either with np.uint8 or np.float32 dtype.
        if self.check_input(im):
            # Numpy random states are the same across all the mutli-processing.
            # In order to maintain the randomness, use the system timestamp to
            # manually set numpy seed every time this function is called.
            self.set_numpy_random_seed()
            im = self.apply(im)
            im = np.clip(im.astype(np.float32), a_min=self.min_value, a_max=self.max_value)
            return im
        else:
            raise ValueError(f'Expect input image to be [3D, 4D]-array, but got {im.ndim}D-array.')


class MultivarGaussianNoise(NoiseAugmentation):
    """Multi-variate Gaussian noise.

    The noise in the channels is dependent.

    Args:
        mean: float, the mean of the noise in each channel.
        cor_var: ndarray, a 3x3 matrix of covariance.
    """
    def __init__(self, mean=0., covar=None):
        super().__init__()
        assert covar is not None
        self.mean = np.array([mean, mean, mean])
        self.cor_var = np.array(covar)

    def apply(self, clean_data, **kwargs):
        shape = clean_data.shape
        noise = np.random.multivariate_normal(self.mean, self.cor_var, size=shape[:-1])
        return clean_data + noise


class ChannelIndependentGaussianNoise(NoiseAugmentation):
    """Channel indenpent Gaussian noise.

    The noise in the channels is independent.

    Args:
        mean: float, the mean of the noise in each channel.
        std: float, standard deviation of the noise.
    """
    def __init__(self, mean=0., std=0.01):
        super(ChannelIndependentGaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def apply(self, clean_data, **kwargs):
        shape = clean_data.shape
        noise = self.std * np.random.randn(*shape) + self.mean
        return clean_data + noise


class GrayscaleGaussianNoise(NoiseAugmentation):
    """Single channel Gaussian noise.

    The noise in the channels is broadcast to all the channels.

    Args:
        mean: float, the mean of the noise in each channel.
        std: float, standard deviation of the noise.
    """
    def __init__(self, mean=0., std=0.01):
        super(GrayscaleGaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def apply(self, clean_data, **kwargs):
        shape = list(clean_data.shape)
        shape[-1] = 1
        noise = self.std * np.random.randn(*shape) + self.mean
        return clean_data + noise


class SaltPepperNoise(NoiseAugmentation):
    """Salt and pepper noise.
    
    Args:
        amount: float, total proportion of the noise pixels in the image.
        salt_ratio: float, the proportion of the salt (white) in the noisy pixels.
    """
    def __init__(self, amount=0.005, salt_ratio=0.5):
        super().__init__()
        self.amount = amount
        self.salt_noise_ratio = salt_ratio

    def apply(self, clean_data, **kwargs):
        h, w = clean_data.shape[:2]
        # make a copy
        noisy = np.array(clean_data)

        num_salt = np.ceil(self.amount * h * w * self.salt_noise_ratio)
        coord = [np.random.randint(0, i - 1, int(num_salt)) for i in [h, w]]
        noisy[tuple(coord)] = self.max_value

        num_pepper = np.ceil(self.amount * h * w * (1. - self.salt_noise_ratio))
        coord = [np.random.randint(0, i - 1, int(num_pepper)) for i in [h, w]]
        noisy[tuple(coord)] = self.min_value
        return noisy


class SpeckleNoise(NoiseAugmentation):
    """Spekle noise.
    """
    def apply(self, clean_data, **kwargs):
        shape = clean_data.shape
        gauss = np.random.randn(*shape)
        noisy = clean_data + clean_data * gauss
        return noisy


class JPEGCompressionNoise(NoiseAugmentation):
    """JPEG compression noise.

    Args:
        quality: int, ranged in [0, 100], controls the quality of the compressed
            image. The lower the quality is, the worse the image looks like.
    """
    def __init__(self, quality):
        super(JPEGCompressionNoise, self).__init__()
        self.quality = int(quality)

    def apply(self, clean_data, **kwargs):
        clean_data = np.clip(clean_data, self.min_value, self.max_value).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        result, encimg = cv2.imencode('.jpg', clean_data, encode_param)
        noisy = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        return noisy.astype(np.float32)
    

class PoissonNoise(NoiseAugmentation):
    """Poisson noise.

    Reference https://stackoverflow.com/questions/19289470/adding-poisson-noise-to-an-image
    https://github.com/xinntao/BasicSR/blob/master/basicsr/data/degradations.py#L587
    """
    def apply(self, clean_data, **kwargs):
        # round and clip image for counting vals correctly
        vals = len(np.unique(clean_data.astype(np.uint8)))
        vals = 2**np.ceil(np.log2(vals))

        img = np.clip(clean_data, self.min_value, self.max_value) / self.max_value
        out = np.float32(np.random.poisson(img * vals) / float(vals))
        noise = (out - img) * self.max_value

        return clean_data + noise
