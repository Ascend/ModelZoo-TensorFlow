#!/usr/bin/env python3
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
# -*- coding: UTF-8 -*-
# Reference: https://zhuanlan.zhihu.com/p/27298134


from npu_bridge.npu_init import *
import os
import caffe
import numpy as np
import pickle


def load_net(proto_path, weights_path):
    """Load caffe model"""
    caffe.set_mode_cpu()
    return caffe.Net(proto_path, weights_path, caffe.TEST)


def print_layer_info(net):
    """Print the names and the shapes of the layers"""
    print('####### print_layer_info #######')

    for layer_name, blob in net.blobs.items():
        print('Shape of layer {:15}: {}'.format(layer_name, blob.data.shape))

    print('\n')


def print_param_info(net):
    """Print the names and parameter shapes of the layers"""
    print('####### print_param_info #######')

    for name, param in net.params.items():
        print('Parameter shape(s) of', name)
        for i in range(len(param)):
            print(param[i].data.shape)

    print('\n')


def dimension_transform(array):
    """
    convert the 4D array order from caffe style to tensorflow style

    caffe blob ordering:        (num, channels, height, width)
    tensorflow weight ordering: (height, width, channels, num)
    """
    # print(array.shape)
    return np.transpose(array, (2, 3, 1, 0))


def load_params(net, tfstyle=True):
    """
    Load the parameter values of the layers

    tfstyle: convert the 4D parameter order to tensorflow style
    """
    params = {}
    for name, param in net.params.items():
        if len(param) == 1:
            params[name + '/kernel'] = dimension_transform(param[0].data) if tfstyle else param[0].data
        elif len(param) == 2:
            if name == 'scale5c_branch2a':
                # in caffe: BatchNorm = BatchNorm Layer + Scale Layer
                # Reference:
                # https://blog.csdn.net/hjxu2016/article/details/81813535
                # https://blog.csdn.net/zziahgf/article/details/78843350
                # etc
                params['bn5c_branch2a' + '/gamma'] = param[0].data
                params['bn5c_branch2a' + '/beta'] = param[1].data
            else:
                params[name + '/weights'] = dimension_transform(param[0].data) if tfstyle else param[0].data
                params[name+'/biases'] = param[1].data
        elif len(param) == 3:
            params[name + '/moving_mean'] = param[0].data / param[2].data
            params[name + '/moving_variance'] = param[1].data / param[2].data

    # print('####### load_params #######')
    # print(params)
    # print('\n')
    return params


def save_params(net, save_path, pkl_name, tfstyle=True):
    print('converting the weights to pickle file...')
    params = load_params(net, tfstyle=tfstyle)
    with open(os.path.join(save_path, pkl_name), 'wb') as f:
        pickle.dump(params, f)
    print('done.')


def caffe2pkl(bpath, prototxt, caffemodel, pkl_name, spath=None):
    # configurations
    BASEPATH = bpath
    PROTOTXT = prototxt
    CAFFEMODEL = caffemodel
    PKL_NAME = pkl_name
    SAVEPATH = BASEPATH if spath is None else spath

    net = load_net(os.path.join(BASEPATH, PROTOTXT), os.path.join(BASEPATH, CAFFEMODEL))
    # print_layer_info(net)
    print_param_info(net)
    save_params(net, SAVEPATH, PKL_NAME, tfstyle=True)
