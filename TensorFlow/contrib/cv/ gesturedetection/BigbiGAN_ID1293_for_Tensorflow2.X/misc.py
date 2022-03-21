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

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np


def get_fixed_random(config, num_to_generate=100):
    seed_cont = tf.random.truncated_normal([num_to_generate, 100])
    seed_cat = tf.math.mod(tf.range(0, num_to_generate), config.num_classes)
    return seed_cont, seed_cat


def generate_images(generator, z_input, c_input, config):
    if not config.conditional:
        c_input = None
    predictions = generator(z_input, c_input, training=False)
    gen_img = _data2plot(predictions, config)

    return gen_img

def _fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 3d numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    canvas = FigureCanvas(fig)
    canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()

    image = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8).reshape(height.astype(np.int32), width.astype(np.int32), 3)

    return image


def _data2plot(array, config):
    fig = plt.figure(figsize=(10, 10))
    if config.dataset in ['mnist', 'fashion_mnist']:  # color channel of the dataset
        for i in range(array.shape[0]):
            plt.subplot(10, 10, i + 1)  # 10*10 subplots
            plt.imshow(array[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
    else:
        for i in range(array.shape[0]):
            plt.subplot(10, 10, i + 1)  # 10*10 subplots
            plt.imshow(array[i, :, :, :])
            plt.axis('off')
    return _fig2data(fig)