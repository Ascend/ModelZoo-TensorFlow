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
# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np
import models.resnet as resnet


class Backbone(tf.keras.layers.Layer):
    def __init__(self, weights=None):
        super(Backbone, self).__init__()
        model = resnet.resnet_18()
        model.build(input_shape=(None, 256,256,3))
        if weights is not None:
            model.load_weights(weights)
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.pool1 = model.pool1
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
        if False:
            self.conv1.trainable = False
            self.bn1.trainable = False
            self.pool1.trainable = False
            
            for layer in self.layer1.layers:
                layer.trainable = False
            
            for layer in self.layer2.layers:
                layer.trainable = False
                
            for layer in self.layer3.layers:
                layer.trainable = False
                
            for layer in self.layer4.layers:
                layer.trainable = False
        
    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x2 = self.layer2(x, training=training)
        x3 = self.layer3(x2, training=training)
        x4 = self.layer4(x3, training=training)
        
        return x2,x3,x4


class ConvBnRelu(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=kernel_size,
                                            strides=stride,
                                            padding='SAME', #padding,
                                           dilation_rate=dilation,
                                           use_bias=bias,
                                           input_shape=(in_channels,))
        self.bn1 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.keras.layers.ReLU()(x) #tf.nn.relu(x)

        return x


class UltraFastNet(tf.keras.Model):
    def __init__(self, num_lanes=4, size=(288, 800), cls_dim=(37, 10, 4), use_aux=False, resnet_weights=None):
        super(UltraFastNet, self).__init__()
        
        self.num_lanes = num_lanes
        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)

        self.backbone = Backbone(weights=resnet_weights)
        
        if self.use_aux:
            self.aux_header2 = tf.keras.Sequential([
                ConvBnRelu(128, 128, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(128,128,3,padding=1),
                ConvBnRelu(128,128,3,padding=1),
                ConvBnRelu(128,128,3,padding=1),
            ], name='aux_header2')
            self.aux_header3 = tf.keras.Sequential([
                ConvBnRelu(256, 128, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
            ], name='aux_header3')
            self.aux_header4 = tf.keras.Sequential([
                ConvBnRelu(512, 128, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(128,128,3,padding=1),
            ], name='aux_header4')
            self.aux_combine = tf.keras.Sequential([
                ConvBnRelu(384, 256, 3, padding=2, dilation=2),
                ConvBnRelu(256, 128, 3, padding=2, dilation=2),
                ConvBnRelu(128, 128, 3, padding=2, dilation=2),
                ConvBnRelu(128, 128, 3, padding=4, dilation=4),
                tf.keras.layers.Conv2D(128,num_lanes + 1,1)
            ], name='aux_combine')
            
        self.cls = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, input_shape=(1800,)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.total_dim, input_shape=(2048,)),
        ], name='cls')
        
        self.pool = tf.keras.layers.Conv2D(8, 1, input_shape=(512,), name='pool')
        
        self.sm = tf.keras.layers.Softmax(axis=1)
        
        self.build((None, self.h, self.w, 3))

    def call(self, inputs, training=None, mask=None):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        #inputs = tf.keras.layers.InputLayer(input_shape=(None, self.h, self.h, 3))
        x2,x3,fea = self.backbone(inputs)
        
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x3)
            x4 = self.aux_header4(fea)
            x4 = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear')(x4)
            aux_seg = tf.keras.layers.Concatenate(axis=1)([x2,x3,x4])
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

        fea = tf.reshape(self.pool(fea), (-1, 1800))

        group_cls = tf.reshape(self.cls(fea), (-1, *self.cls_dim))
        
        group_cls_sm = self.sm(group_cls)

        if self.use_aux:
            return group_cls_sm, aux_seg

        return group_cls_sm