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

from npu_bridge.npu_init import *
import tensorflow as tf
from tensorflow.contrib import layers
import sys

class basic_network(object):
    def __init__(self, cfg):
        self.training=True
        self.cfg = cfg
        self.params_count = 0 #the amount of parameters
    def init_params(self, *args, **kwargs):
        def _variable_on_cpu(w_shape, b_shape, weight_decay=0.99, use_bias=True, name="conv"):
            with tf.device('/cpu:0'):
                w = tf.Variable(tf.truncated_normal(w_shape, 0.0, 0.001), trainable=True, name="%s_w" % name)
                tf.add_to_collection(name="weights_l2_loss", value=self.calc_l1_loss(w, weight_decay))
                b = tf.Variable(tf.zeros(b_shape), trainable=use_bias, name="%s_b" % name)
            return w, b   
        kernel_size = kwargs["kernel_size"]
        in_channels = kwargs["in_channels"]
        out_channels = kwargs["out_channels"]
        # weight_decay = kwargs["weight_decay"]
        w_shape = [kernel_size, kernel_size, in_channels, out_channels]
        b_shape = [out_channels]
        name = kwargs["name"]
        self.params_count += kernel_size*kernel_size*in_channels*out_channels
        self.params_count += out_channels
        return _variable_on_cpu(w_shape, b_shape, use_bias=kwargs["use_bias"], name=name)
        
    def calc_loss(self, *args, **kwargs):
        loss_type = kwargs["loss_type"]
        x = kwargs["x"]
        y = kwargs["y"]
        if loss_type == "L1":
            return tf.reduce_sum(tf.abs(x-y), name="L1_loss")
        elif loss_type == "L2":
            return tf.nn.l2_loss((x-y), name="L2_loss")
        
    def activation(self, *args, **kwargs):
        act_type = kwargs["act_type"]
        act_type = act_type.lower()
        if act_type == "relu":
            return tf.nn.relu(args[0])
        elif act_type == "lrelu":
            slope = kwargs["slope"]
            y = slope*args[0]
            return tf.maximum(args[0], y)
        elif act_type == "prelu":
            return tf.nn.leaky_relu(args[0], alpha=0.2)
        elif act_type == "tanh":
            return tf.nn.tanh(args[0])
        else:
            return args[0]
        
    def calc_l2_loss(self, weight, weight_decay):
        _, _, _, outchannel = weight.get_shape().as_list()
        return (weight_decay) * tf.reduce_sum(tf.square(weight)) / outchannel
        
    def calc_l1_loss(self, weight, weight_decay):
        _, _, _, outchannel = weight.get_shape().as_list()
        return (weight_decay)*tf.reduce_sum(tf.abs(weight)) / outchannel
    
    def batch_norm(self, *args, **kwargs):
        return tf.layers.batch_normalization(args[0], training=kwargs["training"])
    
    def instance_norm(self, *args, **kwargs):
        return layers.instance_norm(args[0], kwargs["name"])
    
    def hard_sigmoid(self, x):
        return tf.nn.relu6((x+3)/6)

    def hard_swish(self, x):
        return x * self.hard_sigmoid(x)
    
    def global_average_pooling(self, x, name="GAP"):
        return tf.reduce_mean(x, axis=[1, 2], keep_dims=True, name="Global_Average_Pooling_%s" % name)#不降维

    
    def ConvBlock(self,x, in_channels, out_channels, kernel_size, stride=1, name="ConvBlock",
                  BN=True, use_bias=True, padding="VALID", act_type="relu", mode="CNA"):

    
        assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!' % sys.modules[__name__]#断言
        weight, bias = self.init_params(kernel_size=kernel_size, in_channels=in_channels,
                                        out_channels=out_channels, use_bias=use_bias, name=name)
        if mode == "CNA":
            x = tf.nn.conv2d(x, filter=weight, strides=[1, stride, stride, 1], padding=padding)
            x = tf.nn.bias_add(x, bias)
            if BN:
                if self.cfg.BN_type == "BN":
                    x = self.batch_norm(x, training=self.cfg.istrain)
                elif self.cfg.BN_type == "IN":
                    x = self.instance_norm(x, name="%s_IN"%name)
                else:
                    raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % self.cfg.BN_type)
            x = self.activation(x, act_type=act_type)
            return x
        elif mode=="NAC":
            if BN:
                if self.cfg.BN_type == "BN":
                    x = self.batch_norm(x, training=self.cfg.istrain)
                elif self.cfg.BN_type == "IN":
                    x = self.instance_norm(x, name="%s_IN" % name)
                else:
                    raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % self.cfg.BN_type)
            x = self.activation(x, act_type=act_type)
            x = tf.nn.conv2d(x, filter=weight, strides=[1, stride, stride, 1], padding=padding)
            x = tf.nn.bias_add(x, bias)
            return x
        
    def DeConvBlock(self, x, in_channels, out_channels, kernel_size, stride=1, name="DeConvBlock",
                    BN=True, use_bias=True, padding="VALID", act_type="relu", mode="CNA"):
        assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!' % sys.modules[__name__]
        b, h, w, c = x.get_shape().as_list()
        out_shape = [b, h * self.cfg.scale, w * self.cfg.scale, out_channels]
        weight, bias = self.init_params(kernel_size=kernel_size, in_channels=out_channels,
                                        out_channels=in_channels, use_bias=use_bias, name=name)
        if mode == "CNA":
            x = tf.nn.conv2d_transpose(x, filter=weight, output_shape=out_shape,
                                       strides=[1, stride, stride, 1], padding=padding)
            x = tf.nn.bias_add(x, bias)
            if BN:
                if self.cfg.BN_type == "BN":
                    x = self.batch_norm(x, training=True)
                elif self.cfg.BN_type == "IN":
                    x = self.instance_norm(x, name="%s_IN" % name)
                else:
                    raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % self.cfg.BN_type)
            x = self.activation(x, act_type=act_type)
            return x
        elif mode == "NAC":
            if BN:
                if self.cfg.BN_type == "BN":
                    x = self.batch_norm(x, training=True)
                elif self.cfg.BN_type == "IN":
                    x = self.instance_norm(x, name="%s_IN" % name)
                else:
                    raise NotImplementedError('[ERROR] BN type [%s] is not implemented!' % self.cfg.BN_type)
            x = self.activation(x, act_type=act_type)
            x = tf.nn.conv2d_transpose(x, filter=weight, output_shape=out_shape,
                                       strides=[1, stride, stride, 1], padding=padding)
            x = tf.nn.bias_add(x, bias)
            return x
