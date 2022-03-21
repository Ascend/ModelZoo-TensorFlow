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
from keras import layers
import keras.backend as K
import keras
from keras.layers import Lambda, concatenate, MaxPooling2D, Concatenate
from data_process.config import classes
from keras import regularizers


class SpaceToDepth(layers.Layer):
    def __init__(self, block_size, **kwargs):
        # self.block_size = block_size
        self.block_size = 2
        super(SpaceToDepth, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        batch, height, width, depth = K.int_shape(x)
        batch = -1
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        y = K.reshape(x, (batch, reduced_height, self.block_size,
                          reduced_width, self.block_size, depth))
        z = K.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
        t = K.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size ** 2))
        return t

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0], input_shape[1] // self.block_size, input_shape[2] // self.block_size,
                 input_shape[3] * self.block_size ** 2)
        return shape


IMAGSZ = 416
input_image = layers.Input((IMAGSZ, IMAGSZ, 3), dtype='float32')  # input layer
x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
x = layers.BatchNormalization(name='norm_1')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# layers2
x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_2')(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# Layer 3
x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_3')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 4
x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_4')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 5
x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_5')(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_6')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 7
x = layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_7')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 8
x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_8')(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Layer 9
x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_9')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 10
x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_10')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 11
x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_11')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 12
x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_12')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 13
x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_13')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

skip_x = x  # [b,32,32,512]
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

print("skip_x.shape")
print(skip_x.shape)
print("x.shape")
print(x.shape)
# for skip connection
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Layer 14
x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_14')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 15
x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_15')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 16
x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_16')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 17
x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_17')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 18
x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False, )(x)
x = layers.BatchNormalization(name='norm_18')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 19
x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_19')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 20
x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_20')(x)
x = layers.LeakyReLU(alpha=0.1)(x)

# Layer 21
skip_x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(skip_x)
skip_x = layers.BatchNormalization(name='norm_21')(skip_x)
skip_x = layers.LeakyReLU(alpha=0.1)(skip_x)

skip_x = SpaceToDepth(block_size=2)(skip_x)
print("x.shape")
print(x.shape)
print("skip_x.shape")
print(skip_x.shape)

# # concat
# # [b,16,16,1024], [b,16,16,256],=> [b,16,16,1280]
x = Concatenate(axis=-1)([skip_x, x])  # Add the number of channels

# Layer 22
x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
x = layers.BatchNormalization(name='norm_22')(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.5)(x)  # add dropout
# [b,16,16,5,7] => [b,16,16,35]
# #layers23
print("len classes")
print(len(classes))
output = layers.Conv2D(5 * (5 + len(classes)), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
# output = layers.Reshape((13,13,5,5+len(classes)))(x)
# create model
model_body = keras.models.Model(input_image, output)
import tensorflow as tf

if __name__ == "__main__":
    # Test model
    x = tf.random.normal((8, 416, 416, 3))
    out = model_body(x)
    print("Model output shape:", out.shape)
    # 打印模型输出
    print("Model structure ========")
    print(model_body.summary())
