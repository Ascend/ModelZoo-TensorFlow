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
import tensorflow.keras.backend as K
from tensorflow.keras import backend 
from tensorflow.keras import layers
from tensorflow.keras import models, Sequential
from tensorflow.keras import utils as keras_utils
from tensorflow.keras.layers import Input
import numpy as np
from tensorflow_core.python.keras.layers import Activation
from tensorflow.keras.models import Model


def get_aggregation_gate(in_filters, reduction=16):
        gate = Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(in_filters // reduction, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dense(in_filters),
            layers.Activation('sigmoid'),
            layers.Reshape((1, 1, -1))      # reshape as (H, W, C)
        ])
        return gate


def construct_fc_layer(feature_dims, input_dim, dropout_p = None, name="features"):
        if feature_dims is None or feature_dims < 0:
            feature_dims =  input_dim
            return None
        
        if isinstance(feature_dims, int):
            feature_dims = [feature_dims]
        
        new_layers = []
        for dim in feature_dims:
            new_layers.append(layers.Dense(dim))
            new_layers.append(layers.BatchNormalization())
            new_layers.append(layers.Activation('relu'))
            if dropout_p is not None:
                new_layers.append(layers.Dropout(dropout_p))
            input_dim = dim
        feature_dims = feature_dims[-1]
        
        return Sequential(new_layers, name=name)
    
class conv2d_bn(tf.keras.Model):
    def __init__(self,
                filters,
                kernel_size = (3, 3),
                padding = 'same',
                strides = (1, 1),
                activation = 'relu',
                **kwargs):
        super(conv2d_bn, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        
        self.bn_axis = -1
        self.conv = layers.Conv2D(self.filters, self.kernel_size, strides=self.strides, padding = self.padding, use_bias=False)
        self.bn_norm = layers.BatchNormalization(axis=self.bn_axis)
        
    def call(self, x):
            x = self.conv(x)
            x = self.bn_norm(x)
            if self.activation is not None:
                x = layers.Activation(self.activation)(x)
            return x

class Light_conv3x3_bn(tf.keras.Model):    
    def __init__(self, filters):
        super(Light_conv3x3_bn, self).__init__()
        self.bn_axis = 3
        self.filters = filters
        
        self.conv = layers.Conv2D(self.filters, kernel_size=1, strides=1, padding='same', use_bias=False)
        self.depth_conv = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn_norm = layers.BatchNormalization(axis=self.bn_axis)
        self.activation = layers.Activation('relu')
        
    def call(self, x):
        x = self.conv(x)
        x = self.depth_conv(x)
        x = self.bn_norm(x)
        x = self.activation(x)
        return x
    
class os_bottleneck(tf.keras.Model):
    def __init__(self,out_filters, bottleneck_reduction=4, **kwargs):
        super(os_bottleneck, self).__init__(**kwargs)
        self.out_filters = out_filters
        self.mid_filters = self.out_filters // bottleneck_reduction
        
        self.conv1 = conv2d_bn(self.mid_filters, kernel_size=(1, 1))
        
        self.conv2a = Light_conv3x3_bn(self.mid_filters)
        self.conv2b = Sequential([
                Light_conv3x3_bn(self.mid_filters),
                Light_conv3x3_bn(self.mid_filters)
        ])
        self.conv2c = Sequential([
                Light_conv3x3_bn(self.mid_filters),
                Light_conv3x3_bn(self.mid_filters),
                Light_conv3x3_bn(self.mid_filters)
        ])
        self.conv2d = Sequential([
                Light_conv3x3_bn(self.mid_filters),
                Light_conv3x3_bn(self.mid_filters),
                Light_conv3x3_bn(self.mid_filters),
                Light_conv3x3_bn(self.mid_filters)
        ])
        
        self.conv3 = conv2d_bn(self.out_filters, kernel_size=(1, 1), activation=None)
        self.ident_conv = conv2d_bn(self.out_filters, kernel_size=(1, 1), activation=None)
        self.gate = get_aggregation_gate(self.mid_filters)
        
        self.Adder = layers.Add()
        self.Multiply = layers.Multiply()
        self.activation = layers.Activation('relu')
        
    def call(self, x):
        in_filters = x.shape[-1]
        identity = x
        
        x1 = self.conv1(x)

        #CONV2a
        branch1 = self.conv2a(x1)

        #CONV2b
        branch2 = self.conv2b(x1)

        #CONV2c
        branch3 = self.conv2c(x1)

        #CONV2d
        branch4 = self.conv2d(x1)

        x2 = self.Adder([
            self.Multiply([branch1, self.gate(branch1)]),
            self.Multiply([branch2, self.gate(branch2)]),
            self.Multiply([branch3, self.gate(branch3)]),
            self.Multiply([branch4, self.gate(branch4)])
        ])

        x3 = self.conv3(x2)
        
        if in_filters != self.out_filters:
            #print("before",identity.shape, in_filters ,self.out_filters) 
            identity = self.ident_conv(identity)
            #print("after",identity.shape)
            
        out = self.Adder([identity, x3]) # residual connection, out = x3 + identity in Pytorch
        out = self.activation(out)
        return out


class OSNet(tf.keras.Model):
    def __init__(self,
                 classes,
                 include_top = False,
                 input_tensor = None,
                 pooling = None,
                 feature_dim = 512,
                 loss_type = {'xent'},
                 **kwargs):
        
        super(OSNet, self).__init__()

        self.include_top = include_top
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.feature_dim = feature_dim
        self.loss_type = loss_type

        #Network component

        self.conv1 = conv2d_bn(64, (7, 7), strides=(2, 2), name="conv1")            # conv1: 128x64x64
        self.maxpool = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="1p") # 1p: 64x32x64

        self.a2 = os_bottleneck(256, name="2a")                               # 2a: 64x32x256
        self.b2 = os_bottleneck(256, name="2b")                               # 2b: 64x32x256
        self.t2 = conv2d_bn(256, (1, 1), name="2t")                                      # 2t: 64x32x256
        self.p2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="2p")    # 2p: 32x16x256

        self.a3 = os_bottleneck(384, name="3a")                               # 3a: 16x8x384
        self.b3 = os_bottleneck(384, name="3b")                               # 3b: 16x8x384
        self.t3 = conv2d_bn(384, (1, 1), name="3t")                                      # 3t: 16x8x384
        self.p3 = layers.AveragePooling2D(pool_size = (2, 2), strides = (2, 2), name="3p") # 3p: 8x4x384

        self.a4 = os_bottleneck(512, name="4a")                               # 4a: 8x4x512
        self.b4 = os_bottleneck(512, name="4b")                               # 4b: 8x4x512
        self.t4 = conv2d_bn(512, (1, 1), name="4t")                                      # 4t: 8x4x512
        
        self.global_avgpool = layers.GlobalAveragePooling2D(name="Avg_pooling")
        self.aft_dropout = layers.Dropout(0.2)
        self.fc = construct_fc_layer(self.feature_dim, 512)
        
        # self.classifier = layers.Dense(self.classes, name="classifier")
        self.activation = layers.Activation("softmax", name="softmax")
        self.model = self.build(Input(shape=(128, 64, 3)), classes)

    def build(self, input_tensor, classes, training=True, return_featuremaps = False):
        """
        self.img_input = layers.Input(tensor=self.input_tensor, shape=self.input_shape)
        if self.input_tensor is None:
            if self.input_shape is None:
                raise ValueError('neither input_tensor nor input_shape is given')
        else:
            if not backend.is_keras_tensor(self.input_tensor):
                self.img_input = layers.Input(tensor=self.input_tensor, shape=self.input_shape)
            else:
                self.img_input = self.input_tensor
        """


        x = self.conv1(input_tensor)
        x = self.maxpool(x)
        
        x = self.a2(x)
        x = self.b2(x)
        x = self.t2(x)
        x = self.p2(x)
        
        x = self.a3(x)
        x = self.b3(x)
        x = self.t3(x)
        x = self.p3(x)
        
        x = self.a4(x)
        x = self.b4(x)
        x = self.t4(x)
        
        if return_featuremaps:
            return x
        
        v = self.global_avgpool(x)
        v = self.aft_dropout(v)

        if self.fc is not None:
            v = self.fc(v)
        # return v

        if not training:
            return v
        y = layers.Dense(classes, name="classifier")(v)
        if self.loss_type == {'xent'}:
            w = self.activation(y)
            # return layers.Model(Input(shape=(128, 64, 3)), output=w, name='osnet')
            return Model(input_tensor, w)
        elif self.loss_type == 'triplet':
            return y, v

        else:
            raise KeyError("Unsupported Loss: {}".format(self.loss))
        
    
if __name__ == "__main__":
    # from keras.utils.vis_utils import plot_model
    # model = OSNet(751)
    # plot_model(model, to_file="model.png", show_shapes=True);


    raw_input = (1, 256, 128, 3)
    model = OSNet(751).model
    # output = model(Input(shape=(256, 128, 3)), training=True)
    print("done")

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    optimizer = tf.keras.optimizers.SGD(lr=0.065, momentum=0.9)
    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = ['accuracy'])
    # print(output)
    # model.build(input_shape=raw_input)
    print(model.summary())
    # https://stackoverflow.com/questions/62242330/error-when-subclassing-the-model-class-you-should-implement-a-call-method

    names = [layer.name for layer in model.layers]
    print(names, len(names))

    # tf.keras.utils.plot_model(
    #         model,                     # here is the trick (for now)
    #         to_file='model.png', dpi=96,              # saving
    #         show_shapes=True, show_layer_names=True,  # show shapes and layer name
    #         expand_nested=True                       # will show nested block
    #     )

    # trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    # print(trainable_count)


