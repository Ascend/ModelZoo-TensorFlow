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

"""
Created on Aug 6th, 2018

This file contains some supporting functions used during training and testing.

@author:Hemant


Migration for Huawei ModelArts finished on 21th Nov, 2021

@contributor: Robert LIU
"""

import os.path

import time
import numpy as np
from scipy.io import loadmat

# If skimage == 0.18.x, then use the following line instead of its next line
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
#from skimage.measure import compare_psnr, compare_ssim


# This provide functionality similar to matlab's tic() and toc()
def TicTocGenerator():
    """
    Generator that returns time differences.
    """
    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator


# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    """
    Prints the time difference yielded by generator instance TicToc.
    """
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():
    """
    Records a time in TicToc, marks the beginning of a time interval.
    """
    toc(False)


# %%
def create_sos(ksp):
    """
    Create SOS (sum of squares).
    """
    nImg, nCh, nrow, ncol = ksp.shape
    scale = np.sqrt(float(nrow) * float(ncol))
    fun = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    for i in range(nImg):
        for j in range(nCh):
            ksp[i, j, :, :] = fun(ksp[i, j, :, :])
    sos = np.sqrt(np.sum(np.square(np.abs(ksp) * scale), axis=1))
    return sos


# %%
def mse(gt, pred):
    """
    Compute Mean Squared Error (MSE).
    """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """
    Compute Normalized Mean Squared Error (NMSE).
    """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """
    Compute Peak Signal to Noise Ratio metric (PSNR).
    """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """
    Compute Structural Similarity Index Metric (SSIM).
    """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )


def ssimch(gt, pred):
    """
    Compute Structural Similarity Index Metric (SSIM).
    """
    return compare_ssim(
        gt, pred, multichannel=True, data_range=gt.max()
    )


# %% Here I am reading the dataset for training and testing from dataset.hdf5 file


def getData(data_path, nImg=360):
    """
    Read and under-sample the raw data for training.
    """
    print('Reading data')
    tic()
    org = np.load(os.path.join(data_path, 'trn_data_90im_4_subjects.npy'))
    mask = loadmat(os.path.join(data_path, 'vardenmask_6f.mat'))['b']
    mask = mask.astype(np.complex64)
    mask = np.tile(mask, [nImg, 1, 1])
    toc()
    print('Undersampling')
    tic()
    orgk, atb, _ = generateUndersampled(org, mask)
    atb = c2r(atb)
    orgk = c2r(orgk)
    mask = np.expand_dims(mask, axis=1)
    mask = np.tile(mask, [1, 12, 1, 1])
    mask = np.expand_dims(mask, axis=-1)
    orgk = np.expand_dims(orgk, axis=-1)
    atb = np.expand_dims(atb, axis=-1)
    toc()
    print('Data prepared!')
    return orgk, atb, mask


def getTestingData(data_path, nImg=1):
    """
    Read and under-sample the raw data for testing.
    """
    print('Reading data')
    tic()
    org = np.load(os.path.join(data_path, 'tst_img.npy'))
    mask = loadmat(os.path.join(data_path, 'vardenmask_6f.mat'))['b']
    mask = mask.astype(np.complex64)
    mask = np.tile(mask, [nImg, 1, 1])
    toc()
    print('Undersampling')
    tic()
    orgk, atb, minv = generateUndersampled(org, mask)
    atb = c2r(atb)
    orgk = c2r(orgk)
    mask = np.expand_dims(mask, axis=1)
    mask = np.tile(mask, [1, 12, 1, 1])
    mask = np.expand_dims(mask, axis=-1)
    orgk = np.expand_dims(orgk, axis=-1)
    atb = np.expand_dims(atb, axis=-1)
    toc()
    print('Data prepared!')
    return {'org': org, 'orgk': orgk, 'atb': atb, 'mask': mask, 'minv': minv}


# %%

def usp(x, mask, nrow, ncol, ncoil):
    """
    This is a the A operator as defined in the paper.
    """
    kspace = np.reshape(x, (ncoil, nrow, ncol))
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    res = kspace[mask != 0]
    return kspace, res


def usph(kspaceUnder, mask, nrow, ncol, ncoil):
    """
    This is a the A^T operator as defined in the paper.
    """
    temp = np.zeros((ncoil, nrow, ncol), dtype=np.complex64)
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    temp[mask != 0] = kspaceUnder
    minv = np.std(temp)
    temp = temp / minv
    return temp, minv


def generateUndersampled(org, mask):
    """
    Generate under-sampled data.
    """
    nSlice, ncoil, nrow, ncol = org.shape
    orgk = np.empty(org.shape, dtype=np.complex64)
    atb = np.empty(org.shape, dtype=np.complex64)
    minv = np.zeros((nSlice,), dtype=np.complex64)
    for i in range(nSlice):
        A = lambda z: usp(z, mask[i], nrow, ncol, ncoil)
        At = lambda z: usph(z, mask[i], nrow, ncol, ncoil)
        orgk[i], y = A(org[i])
        atb[i], minv[i] = At(y)
        orgk[i] = orgk[i] / minv[i]
    del org
    return orgk, atb, minv


# %%
def r2c(inp):
    """
    Input image: row x col x 2 in float32
    Output image: row  x col in complex64
    """
    if inp.dtype == 'float32':
        dtype = np.complex64
    else:
        dtype = np.complex128
    inp = np.squeeze(inp, axis=-1)
    nImg, nCh, nrow, ncol = inp.shape
    out = np.zeros((nImg, nCh, nrow, ncol), dtype=dtype)
    re, im = np.split(inp, 2, axis=1)
    out = re + (1j * im)
    return out


def c2r(inp):
    """
    Input image: row x col in complex64
    Output image: row  x col x2 in float32
    """
    if inp.dtype == 'complex64':
        dtype = np.float32
    else:
        dtype = np.float64
    nImg, nCh, nrow, ncol = inp.shape
    out = np.zeros((nImg, nCh * 2, nrow, ncol), dtype=dtype)
    out[:, 0:nCh, :, :] = np.real(inp)
    out[:, nCh:nCh * 2, :, :] = np.imag(inp)
    return out
