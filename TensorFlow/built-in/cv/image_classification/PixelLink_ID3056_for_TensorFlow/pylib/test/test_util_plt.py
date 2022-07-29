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
from npu_bridge.npu_init import *
import numpy as np

import matplotlib.pyplot as plt
import util

def test_to_ROI():
    image_path = '~/dataset/ICDAR2015/Challenge4/ch4_training_images/img_150.jpg'
    image = util.img.imread(image_path, rgb = True)
    ax = plt.subplot(111)
    ax.imshow(image)
    ROI = [(53, 113), (377, 275)]
    util.plt.to_ROI(ax, ROI)
    plt.show()


@util.dec.print_test
def test_save_images():
    path = '~/temp/result/test.png'
    shape = (100, 100, 3)
    white = util.img.white(shape)
    black = util.img.black(shape)
    util.plt.show_images(images = [white, black], titles = ['white', 'black'], show = False, save = True, path = path, axis_off = True)
    #util.plt.save_images(path = path)

test_save_images()

#test_to_ROI()

