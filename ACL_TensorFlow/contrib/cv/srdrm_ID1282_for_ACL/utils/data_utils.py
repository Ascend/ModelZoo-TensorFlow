#!/usr/bin/env python
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
# > Various modules for handling data 
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
from __future__ import division
from __future__ import absolute_import
import os
import random
import fnmatch
import numpy as np
from scipy import misc
from PIL import Image
from glob import glob


def deprocess(x):
    """
    param: x
    return: x_
    """
    # [-1,1] -> [0, 1]
    return (x + 1.0) * 0.5


def preprocess(x):
    """
    param: x numpy
    return: x_ numpy
    """
    # [0,255] -> [-1, 1]
    return (x / 127.5) - 1.0


def getPaths(data_dir):
    """
    param: data_dir
    return: image_paths numpy
    """
    exts = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.JPEG']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if fnmatch.fnmatch(filename, pattern):
                    fname_ = os.path.join(d, filename)
                    image_paths.append(fname_)
    return np.asarray(image_paths)
