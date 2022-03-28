#!/usr/bin/env bash
#
# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
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
echo "Downloading SVHN"
mkdir -p data/images/svhn
(
    cd data/images/svhn
    curl -O 'http://ufldl.stanford.edu/housenumbers/{train,test,extra}_32x32.mat'
)

echo
echo "Downloading CIFAR-10"
mkdir -p data/images/cifar/cifar10
(
    cd data/images/cifar/cifar10
    curl -O 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz'
    tar xvzf cifar-10-matlab.tar.gz
    mv cifar-10-batches-mat/* .
    rmdir cifar-10-batches-mat
)

echo
echo "Preprocessing CIFAR-10"
python3 datasets/preprocess_cifar10.py

echo
echo "All done!"
