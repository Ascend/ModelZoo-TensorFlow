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
import time
import random
from scipy import special

from src.utils.degradation.base import Base


class BlurKernel2D(Base):
    """A common base class for blurness.
    """
    def check_kernel(self):
        if self.ksize[0] % 2 == 0 or self.ksize[1] % 2 == 0:
            raise ValueError(f'Expect kernel size to be odd (1, 3, 5, etc.), but got {self.ksize}')


class AvgBlur2D(BlurKernel2D):
    """A 2D average blur operator.

    Args:
        k_size: int, blur kernel size, expected to be odd.
    """
    def __init__(self, k_size):
        assert k_size % 2 == 1
        k_size = (k_size, k_size)
        self.k_size = k_size

    def apply(self, im):
        return cv2.blur(im, self.k_size, borderType=cv2.BORDER_REFLECT_101)


class IsotropicGaussianBlur2D(BlurKernel2D):
    """An isotropic Gaussian blur operator.

    Args:
        kernel_size: int, blur kernel size, expected to be odd.
        std: float, width of the kernel.
    """
    def __init__(self, kernel_size, std):
        assert kernel_size % 2 == 1
        self.kernel_size = (kernel_size, kernel_size)
        self.std = std

    def apply(self, im):
        if self.check_input(im):
            return cv2.GaussianBlur(im, self.kernel_size, self.std, self.std, borderType=cv2.BORDER_REFLECT_101)
        else:
            raise ValueError


def gaussian(x, k, s):
    return np.exp(-(x-(k-1)/2)**2/(2*s**2))



class AnisotropicGaussianBlur2D(BlurKernel2D):
    """An anisotropic Gaussian blur operator.
    
    Reference to
    https://github.com/cszn/USRNet/blob/4fb56deb80d655abb722ff83750ad3df163ef833/utils/utils_sisr.py#L129

    Args:
        kernel_size: int, blur kernel size, expected to be odd.
        var: float, width of the kernel.

    """
    def __init__(self, kernel_size, var, angle, scale=1, noise_level=0):
        """"
        # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
        # Kai Zhang
        # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
        # max_var = 2.5 * sf
        """
        assert kernel_size % 2 == 1
        k_size = np.array([kernel_size, kernel_size])

        # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
        lambda_1, lambda_2 = var
        # noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

        # Set COV matrix using Lambdas and Theta
        LAMBDA = np.diag([lambda_1, lambda_2])
        Q = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        SIGMA = Q @ LAMBDA @ Q.T
        INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

        # Set expectation position (shifting kernel for aligned image)
        MU = k_size // 2 - 0.5*(scale - 1)
        MU = MU[None, None, :, None]

        # Create meshgrid for Gaussian
        [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
        Z = np.stack([X, Y], 2)[:, :, :, None]

        # Calcualte Gaussian for every pixel of the kernel
        ZZ = Z-MU
        ZZ_t = ZZ.transpose(0,1,3,2)
        raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ))

        # Normalize the kernel and return
        self.kernel = raw_kernel / np.sum(raw_kernel)

    def apply(self, im):
        return cv2.filter2D(im, -1, self.kernel)


def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """2D sinc filter.
    
    Borrowed from https://github.com/xinntao/BasicSR/blob/master/basicsr/data/degradations.py#L392
    Ref: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
    
    Args:
        cutoff: float, cutoff frequency in radians (pi is max)
        kernel_size: int, horizontal and vertical size, must be odd.
        pad_to: int, pad kernel size to desired size, must be odd or zero.
    
    Returns:
        ndarray of [kernel_size, kernel_size], the sinc kernel.
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


class SincFilter(BlurKernel2D):
    """A sinc filter.

    Args:
        kernel_size: int, blur kernel size, expected to be odd.
        omega_c: float, cutoff frequency in radians (pi is max)
    """
    def __init__(self, kernel_size, omega_c):
        self.kernel_size = kernel_size
        self.omega_c = omega_c
        self.kernel = circular_lowpass_kernel(self.omega_c, self.kernel_size, pad_to=False)

    def apply(self, im):
        return cv2.filter2D(im, -1, self.kernel)
