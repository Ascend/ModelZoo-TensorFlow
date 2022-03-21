# MIT License

# Copyright (c) 2018 Deniz Engin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
import cv2
import os
from skimage.measure import compare_ssim

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model_predict_dir', 'data/testData/model_predict',
                       'model predict output directory, default: data/testData/model_predict')
tf.flags.DEFINE_string('groundtruth_dir', 'data/testData/groundtruth',
                       'groundtruth directory, default: data/testData/groundtruth')


def read_img(path):
    return tf.image.decode_image(tf.io.read_file(path))


def psnr_read_directory(directory_name):
    array_of_img = []
    for filename in os.listdir(r"./"+directory_name):
        img = read_img(directory_name + "/" + filename)
        array_of_img.append(img)
    return array_of_img


def cal_average_psnr():
    array_of_predict_img = psnr_read_directory(FLAGS.model_predict_dir)
    array_of_groundtruth_img = psnr_read_directory(FLAGS.groundtruth_dir)
    img_num = len(array_of_predict_img)
    psnr_sum = 0
    for i in range(img_num):
        temp = tf.image.psnr(array_of_predict_img[i], array_of_groundtruth_img[i], max_val=255)
        psnr_sum += temp.eval()
    print("average psnr is %f" % (psnr_sum/img_num))


def ssim_read_directory(directory_name):
    array_of_img = []
    for filename in os.listdir(r"./"+directory_name):
        img = read_img(directory_name + "/" + filename)
        array_of_img.append(img)
    return array_of_img


def cal_average_ssim():
    array_of_predict_img = ssim_read_directory(FLAGS.model_predict_dir)
    array_of_groundtruth_img = ssim_read_directory(FLAGS.groundtruth_dir)
    img_num = len(array_of_predict_img)
    ssim_sum = 0
    for i in range(img_num):
        temp = tf.image.ssim(array_of_predict_img[i], array_of_groundtruth_img[i], max_val=255)
        ssim_sum += temp.eval()
    print("average ssim is %f" % (ssim_sum/img_num))


def cal_metrics():
    sess = tf.InteractiveSession()
    cal_average_psnr()
    cal_average_ssim()
    sess.close()


if __name__ == '__main__':
    cal_metrics()