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
# Copyright 2020 Huawei Technologies Co., Ltd
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
from tensorflow.layers import Dense, Layer, InputSpec
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.init_ops import RandomNormal as TFRandomNormal
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops


class Attention_Layer(Layer):
    def __init__(self, att_hidden_units, activation='prelu'):
        """
        """
        super(Attention_Layer, self).__init__()
        self.att_dense = [Dense(unit, activation=activation) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

    def call(self, inputs, *args, **kwargs):
        # query: candidate item  (None, d * 2), d is the dimension of embedding
        # key: hist items  (None, seq_len, d * 2)
        # value: hist items  (None, seq_len, d * 2)
        # mask: (None, seq_len)
        q, k, v, mask = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])  # (None, seq_len * d * 2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])  # (None, seq_len, d * 2)

        # q, k, out product should concat
        info = tf.concat([q, k, q - k, q * k], axis=-1)

        # dense
        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)  # (None, seq_len, 1)
        outputs = tf.squeeze(outputs, axis=-1)  # (None, seq_len)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # (None, seq_len)
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)  # (None, seq_len)

        # softmax
        outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len)
        outputs = tf.expand_dims(outputs, axis=1)  # None, 1, seq_len)

        outputs = tf.matmul(outputs, v)  # (None, 1, d * 2)
        outputs = tf.squeeze(outputs, axis=1)

        return outputs


class Dice(Layer):
    def call(self, x, *args, **kwargs):

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            alphas = tf.get_variable('alpha', x.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            input_shape = list(x.get_shape())

            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[-1]
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[-1] = input_shape[-1]

        # case: train mode (uses stats of the current batch)
        mean = tf.reduce_mean(x, axis=reduction_axes)
        brodcast_mean = tf.reshape(mean, broadcast_shape)
        std = tf.reduce_mean(tf.square(x - brodcast_mean) + 1e-8, axis=reduction_axes)
        std = tf.sqrt(std)
        brodcast_std = tf.reshape(std, broadcast_shape)
        x_normed = (x - brodcast_mean) / (brodcast_std + 1e-8)
        x_p = tf.sigmoid(x_normed)

        return alphas * (1.0 - x_p) * x + x_p * x


class RandomNormal(TFRandomNormal):
    """Initializer that generates tensors with a normal distribution.

    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values to
        generate. Defaults to 0.
      stddev: a python scalar or a scalar tensor. Standard deviation of the random
        values to generate. Defaults to 0.05.
      seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed` for behavior.
      dtype: The data type. Only floating point types are supported.

    Returns:
        RandomNormal instance.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=dtypes.float32):
        super(RandomNormal, self).__init__(
            mean=mean, stddev=stddev, seed=seed, dtype=dtype)


class Embedding(Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer=None,
                 embeddings_regularizer=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        dtype = kwargs.pop('dtype', 'float32')

        kwargs['autocast'] = False
        super(Embedding, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.input_length = input_length

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None

        return math_ops.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        if self.input_length is None:
            return input_shape + (self.output_dim,)
        else:
            # input_length can be tuple if input is 3D or higher
            if isinstance(self.input_length, (list, tuple)):
                in_lens = list(self.input_length)
            else:
                in_lens = [self.input_length]
            if len(in_lens) != len(input_shape) - 1:
                raise ValueError('"input_length" is %s, '
                                 'but received input has shape %s' % (str(
                    self.input_length), str(input_shape)))
            else:
                for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
                    if s1 is not None and s2 is not None and s1 != s2:
                        raise ValueError('"input_length" is %s, '
                                         'but received input has shape %s' % (str(
                            self.input_length), str(input_shape)))
                    elif s1 is None:
                        in_lens[i] = s2
            return (input_shape[0],) + tuple(in_lens) + (self.output_dim,)

    def call(self, inputs, **kwargs):
        dtype = inputs.dtype.base_dtype.name
        if dtype != 'int32' and dtype != 'int64':
            inputs = math_ops.cast(inputs, 'int32')
        out = embedding_ops.embedding_lookup(self.embeddings, inputs)
        return out


class PReLU(Layer):

    def __init__(self,
                 alpha_initializer='zeros',
                 alpha_regularizer=None,
                 alpha_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = alpha_initializer
        self.alpha_regularizer = alpha_regularizer
        self.alpha_constraint = alpha_constraint
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        self.alpha = self.add_weight(
            shape=param_shape,
            name='alpha',
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint)
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs, **kwargs):
        pos = tf.nn.relu(inputs)
        neg = -self.alpha * tf.nn.relu(-inputs)
        return pos + neg

    def get_config(self):
        config = {
            'alpha_initializer': self.alpha_initializer,
            'alpha_regularizer': self.alpha_regularizer,
            'alpha_constraint': self.alpha_constraint,
            'shared_axes': self.shared_axes
        }
        base_config = super(PReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
