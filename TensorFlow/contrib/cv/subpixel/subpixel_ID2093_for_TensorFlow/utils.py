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

import cv2
import numpy as np
import tensorflow as tf
import os
import glob
import h5py


# Get the Image
def imread(path, config):
    """
        Args:
        path: the image directory path
        config: the configuration
    """
    img = cv2.imread(path)
    if config.c_dim == 3:
        return img
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        return ycrcb[:, :, 0]


def imsave(image, path, config):
    """
        Args:
        image: the output image
        path: the image directory path
        config: the configuration
    """
    # checkimage(image)
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join('./', config.result_dir)):
        os.makedirs(os.path.join('./', config.result_dir))

    # NOTE: because normial, we need mutlify 255 back
    cv2.imwrite(os.path.join('./', path), image * 255.)


def checkimage(image):
    """
    show image to check
    """
    cv2.imshow("test", image)
    cv2.waitKey(0)


def modcrop(img, scale=3):
    """
        To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    """
    # Check the image is grayscale
    if len(img.shape) == 3:
        h, w, _ = img.shape
        h = int(h / scale) * scale
        w = int(w / scale) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = int(h / scale) * scale
        w = int(w / scale) * scale
        img = img[0:h, 0:w]
    return img


def checkpoint_dir(config,savepath):
    """
    to convert data to h5
        Args:
        config: the configuration
    """
    if config.is_train:
        # print("--------------------------------------config.is_train",config.is_train)
        # print("-------------------config.checkpoint_dir",config.checkpoint_dir)
        #return os.path.join(config.checkpoint_dir, "train.h5")
        return savepath
    else:
        return os.path.join(config.checkpoint_dir, "test.h5")


def preprocess(path, config):
    """
        Args:
            path: the image directory path
            config: the configuration
    """
    img = imread(path, config)

    label_ = modcrop(img, config.scale)

    input_ = cv2.resize(label_, None, fx=1.0 / config.scale, fy=1.0 / config.scale,
                        interpolation=cv2.INTER_CUBIC)  # Resize by scaling factor

    kernel_size = (7, 7)

    sigma = 3.0
    # input_ = cv2.GaussianBlur(input_, kernel_size, sigma)
    # checkimage(input_)

    return input_, img


def prepare_data(dataset="Train", Input_img="", config=None):
    """
        Args:
            dataset: choose train dataset or test dataset
            Input_img: dir of input img
            config: the configuration
            For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp',..., 't99.bmp']
    """
    if dataset == "Train":
        # Join the Train dir to current directory
        # data_dir = os.path.join(os.getcwd(), dataset)  #add
        # print(data_dir)
        data_dir = config.data_dir + dataset
        # print("1111111111",data_dir)
        # make set of all dataset file path
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    else:
        if Input_img != "":
            data = [os.path.join('./', Input_img)]
        else:
            data_dir = os.path.join(os.path.join(config.data_dir, dataset), "Set5")  #pass
            # make set of all dataset file path
            data = glob.glob(os.path.join(data_dir, "*.bmp"))
    print(data)
    return data


def load_data(config):
    """
        Args:
        config: the configuration
    """
    if config.is_train:
        data = prepare_data(dataset="Train", config=config)
    else:
        if config.test_img != "":
            return prepare_data(dataset="Test", Input_img=config.test_img, config=config)
        data = prepare_data(dataset="Test", config=config)
    return data


def make_sub_data(data, config):
    """
        Make the sub_data set
        Args:
            data : the set of all file path
            config : the all flags
    """
    sub_input_sequence = []
    sub_label_sequence = []
    for i in range(len(data)):
        if not config.is_train:
            img = imread(data[i], config)
            label_ = modcrop(img, config.scale)
            cv2.imwrite("oriimg.bmp", img)
            print('dfsfgvs')

            input_ = cv2.resize(label_, None, fx=1.0 / config.scale, fy=1.0 / config.scale,
                                interpolation=cv2.INTER_CUBIC)  # Resize by scaling factor
            input_ = input_ / 255.0
            label_ = label_ / 255.0
            sub_input_sequence.append(input_)
            sub_label_sequence.append(label_)
            continue

        input_, label_, = preprocess(data[i], config)  # do bicbuic
        if len(input_.shape) == 3:  # is color
            h, w, c = input_.shape
        else:
            h, w = input_.shape  # is grayscale

        # NOTE: make subimage of LR and HR

        # Input
        for x in range(0, h - config.image_size + 1, config.stride):
            for y in range(0, w - config.image_size + 1, config.stride):
                sub_input = input_[x: x + config.image_size,
                            y: y + config.image_size]  # 17 * 17

                # Reshape the subinput and sublabel
                sub_input = sub_input.reshape(
                    [config.image_size, config.image_size, config.c_dim])

                # Normialize
                sub_input = sub_input / 255.0

                # Add to sequence
                sub_input_sequence.append(sub_input)

        # Label (the time of scale)
        for x in range(0, h * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
            for y in range(0, w * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
                sub_label = label_[x: x + config.image_size * config.scale,
                            y: y + config.image_size * config.scale]  # 17r * 17r

                # Reshape the subinput and sublabel
                sub_label = sub_label.reshape(
                    [config.image_size * config.scale, config.image_size * config.scale, config.c_dim])
                # Normialize
                sub_label = sub_label / 255.0
                # Add to sequence
                sub_label_sequence.append(sub_label)

    return sub_input_sequence, sub_label_sequence


def read_data(path):
    """
        Read h5 format data file

        Args:
            path: file path of desired file
            data: '.h5' file format that contains  input values
            label: '.h5' file format that contains label values 
    """
    with h5py.File(path, 'r') as hf:
        input_ = np.array(hf.get('input'))
        label_ = np.array(hf.get('label'))
        return input_, label_


def make_data_hf(input_, label_, config):
    """
        Make input data as h5 file format
        Depending on "is_train" (flag value), savepath would be change.
    """
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(config.data_dir, config.checkpoint_dir)):
        os.makedirs(os.path.join(config.data_dir, config.checkpoint_dir))

    if config.is_train:
        savepath = os.path.join(
            config.data_dir, config.checkpoint_dir + '/train.h5')
    else:
        savepath = os.path.join(
            config.data_dir, config.checkpoint_dir + '/test.h5')
    # print("------------------------------------------------------------------savepath",savepath)
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('input', data=input_)
        hf.create_dataset('label', data=label_)
    return savepath


def input_setup(config):
    """
        Read image files and make their sub-images and saved them as a h5 file format
        config: the configuration
    """

    # Load data path, if is_train False, get test data
    data = load_data(config)

    # Make sub_input and sub_label, if is_train false more return nx, ny
    sub_input_sequence, sub_label_sequence = make_sub_data(data, config)

    # Make list to numpy array. With this transform
    # print(sub_input_sequence,sub_input_sequence.shape)
    arrinput = np.asarray(sub_input_sequence)  # [?, 17, 17, 3]
    print("type of input data:", arrinput.dtype)
    # [?, 17 * scale , 17 * scale, 3]
    arrlabel = np.asarray(sub_label_sequence)

    print(arrinput.shape, arrinput.dtype)
    savepath = make_data_hf(arrinput, arrlabel, config)
    return savepath
