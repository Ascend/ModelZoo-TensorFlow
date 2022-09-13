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
"""Transform .jpeg picture to bin file."""

import numpy as np
from PIL import Image as im
import os
import tensorflow as tf

jpegpath = "./jpeg/"       # 最后要加 "/"
output = "./bininput/"    # 最后要加 "/"
if not os.path.exists(output):
    os.makedirs(output)

MEAN_RGB = [127.0, 127.0, 127.0]
STDDEV_RGB = [128.0, 128.0, 128.0]

dirs = [dir for dir in os.listdir(jpegpath) if os.path.isdir(jpegpath + dir)]
for dir in dirs:

    files = [f for f in os.listdir(jpegpath + dir) if f.endswith('.JPEG')]
    for i in files:
        image = im.open(jpegpath + dir + "/" + i)
        image = image.convert("RGB")
        image = image.resize((224, 224), resample=im.BILINEAR)

        image_arr = np.array(image).astype("float32")

        image_arr -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype="float32")
        image_arr /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype="float32")
        image_arr = tf.Session().run(image_arr)

        image_arr.tofile(output + "{}.bin".format(dir.zfill(3)+i[14:-5]))
