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


def spectral_normalization(name, weight, n_itr=1, update_collection=None):
    """
    Args:
        weight: shape ->   fc: [in_dim, out_dim]
                            conv: [h, w, c_in, c_out]
    """
    w_shape = weight.shape.as_list()
    weight = tf.reshape(weight, [-1, w_shape[-1]])      # treat conv weight as a 2-D matrix: [h*w*c_in, c_out]

    # power iteration method
    u = tf.get_variable(name + 'u', [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(),
                        trainable=False)
    u_hat = u       # right singular vector
    v_hat = None    # left singular vector
    # Because the weights change slowly, we only need to perform a single power iteration
    # on the current version of these vectors for each step of learning
    for _ in range(n_itr):
        v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(weight)))
        u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, weight))

    # spectral normalization
    sigma = tf.squeeze(tf.matmul(tf.matmul(v_hat, weight), tf.transpose(u_hat)))
    weight /= sigma

    if update_collection is None:
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(weight, w_shape)            # get original shape
    else:
        w_norm = tf.reshape(weight, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_hat))

    return w_norm


def conv(name, inputs, nums_out, k_size, strides, update_collection=None, is_sn=False):
    """convolution layer (with spectral normalization)"""
    nums_in = inputs.shape[-1]             # num of input channels
    with tf.variable_scope(name):
        w = tf.get_variable("w", [k_size, k_size, nums_in, nums_out], initializer=tf.orthogonal_initializer())
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.0]))
        if is_sn:
            w = spectral_normalization("sn", w, update_collection=update_collection)
        op = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding="SAME")
    return tf.nn.bias_add(op, b)


def dense(name, inputs, nums_out, update_collection=None, is_sn=False):
    """fully connected layer (with spectral normalization)"""
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        w = tf.get_variable("w", [nums_in, nums_out], initializer=tf.orthogonal_initializer())
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.0]))
        if is_sn:
            w = spectral_normalization("sn", w, update_collection=update_collection)
    return tf.nn.bias_add(tf.matmul(inputs, w), b)


def conditional_batchnorm(x, train_phase, name, split_z=None, embed_y=None):
    """implementation of shared embedding and skip-z in the BigGAN paper

    Args:
        split_z: vector -> one chunk of the noise vector "z"
        embed_y: class info (shared embedding)
    """
    with tf.variable_scope(name):
        epsilon = 1e-5          # variance epsilon for batch norm
        decay = 0.9             # decay rate for exponential moving average in batch norm

        if embed_y is None:
            # batch normalization
            beta = tf.get_variable(name=name + 'beta', shape=[x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)
            gamma = tf.get_variable(name=name + 'gamma', shape=[x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)
        else:
            # conditional batch normalization
            z = tf.concat([split_z, embed_y], axis=1)         # get conditional vector
            # use conditional vector to get batchNorm gains and biases
            gamma = dense("gamma", z, x.shape[-1], is_sn=True)      # scale
            beta = dense("beta", z, x.shape[-1], is_sn=True)        # offset
            gamma = tf.reshape(gamma, [-1, 1, 1, x.shape[-1]])
            beta = tf.reshape(beta, [-1, 1, 1, x.shape[-1]])

        # calculate batch mean and variance
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments', keep_dims=True)

        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
    return normed


def down_sampling(inputs):
    """down-sampling: avg pool with zero-padding (out_size = in_size / 2)
    """
    return tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def up_sampling(inputs):
    """nearest-neighbors up-sampling (out_size = in_size * 2)
    """
    h, w = inputs.shape[1], inputs.shape[2]
    return tf.image.resize_nearest_neighbor(inputs, [h * 2, w * 2])


def non_local(name, inputs, update_collection, is_sn):
    """attention module

        This implementation is different from the bigGAN paper. Please check this paper: Non-local Neural Networks.
    It also uses down sampling to reduce computation.
    """
    h, w, num_channels = inputs.shape[1], inputs.shape[2], inputs.shape[3]
    location_num = h * w
    down_sampled_num = location_num // 4    # after down sampling, feature map shrinks to a quarter of its size

    with tf.variable_scope(name):
        # theta: [h*w, c//8]
        theta = conv("f", inputs, num_channels // 8, 1, 1, update_collection, is_sn)
        theta = tf.reshape(theta, [-1, location_num, num_channels // 8])
        # phi: [d_h*d_w, c//8]
        phi = conv("h", inputs, num_channels // 8, 1, 1, update_collection, is_sn)
        phi = down_sampling(phi)
        phi = tf.reshape(phi, [-1, down_sampled_num, num_channels // 8])
        # attention map: [h*w, d_h*d_w]
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)
        # g: [d_h*d_w, c//2]
        g = conv("g", inputs, num_channels // 2, 1, 1, update_collection, is_sn)
        g = down_sampling(g)
        g = tf.reshape(g, [-1, down_sampled_num, num_channels // 2])
        # attn_g: [h*w, c//2]
        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [-1, h, w, num_channels // 2])
        # attn_g: [h*w, c]
        attn_g = conv("attn", attn_g, num_channels, 1, 1, update_collection, is_sn)

        sigma = tf.get_variable("sigma_ratio", [], initializer=tf.constant_initializer(0.0))
        return inputs + sigma * attn_g


def non_local_bigGAN(name, inputs, update_collection, is_sn):
    """attention module

        This implementation follows the bigGAN paper.
    """
    H = inputs.shape[1]
    W = inputs.shape[2]
    C = inputs.shape[3]
    C_ = C // 8
    inputs_ = tf.transpose(inputs, perm=[0, 3, 1, 2])
    inputs_ = tf.reshape(inputs_, [-1, C, H * W])
    with tf.variable_scope(name):
        f = conv("f", inputs, C_, 1, 1, update_collection, is_sn)     # key
        g = conv("g", inputs, C_, 1, 1, update_collection, is_sn)     # query
        h = conv("h", inputs, C, 1, 1, update_collection, is_sn)      # value
        f = tf.transpose(f, [0, 3, 1, 2])
        f = tf.reshape(f, [-1, C_, H * W])
        g = tf.transpose(g, [0, 3, 1, 2])
        g = tf.reshape(g, [-1, C_, H * W])
        h = tf.transpose(h, [0, 3, 1, 2])
        h = tf.reshape(h, [-1, C, H * W])
        # attention map
        s = tf.matmul(f, g, transpose_a=True)
        beta = tf.nn.softmax(s, dim=0)
        o = tf.matmul(h, beta)
        gamma = tf.get_variable("gamma", [], initializer=tf.constant_initializer(0.))
        y = gamma * o + inputs_
        y = tf.reshape(y, [-1, C, H, W])
        y = tf.transpose(y, perm=[0, 2, 3, 1])
    return y


def global_sum_pooling(inputs):
    """global sum pooling

    Args:
        inputs -> shape: [N, H, W, C]

    Returns:
        shape: [N, C]
    """
    return tf.reduce_sum(inputs, axis=[1, 2], keep_dims=False)


def Hinge_loss(real_logits, fake_logits):
    d_loss = -tf.reduce_mean(tf.minimum(0., -1.0 + real_logits)) - tf.reduce_mean(tf.minimum(0., -1.0 - fake_logits))
    g_loss = -tf.reduce_mean(fake_logits)
    return d_loss, g_loss


def ortho_reg(vars_list):
    """apply orthogonal regularization to convolutional layers
    """
    s = 0
    for var in vars_list:
        if "w" in var.name and var.shape.__len__() == 4:
            # w shape: [k_size, k_size, in_channels, out_channels]
            nums_kernel = int(var.shape[-1])
            w = tf.transpose(var, perm=[3, 0, 1, 2])    # [out_channels, k_size, k_size, in_channels]
            w = tf.reshape(w, [nums_kernel, -1])        # [out_channels, k_size*k_size*in_channels]
            ones = tf.ones([nums_kernel, nums_kernel])
            eyes = tf.eye(nums_kernel, nums_kernel)
            y = tf.matmul(w, w, transpose_b=True) * (ones - eyes)
            s += tf.nn.l2_loss(y)
    return s


def d_projection(global_pooled, y, nums_class, update_collection=None):
    """paper: cGANs with Projection Discriminator

    Args:
        global_pooled: hidden layer after global sum pooling. shape -> [N, C]
        y: class info (a scalar, not one-hot encoding!)
        nums_class: number of classes
    """
    w = global_pooled.shape[-1]
    v = tf.get_variable("v", [nums_class, w], initializer=tf.orthogonal_initializer())
    v = tf.transpose(v)
    # V^T acts like a fully connected layer, so we need to perform spectral norm on V^T instead of V
    v = spectral_normalization("embed", v, update_collection=update_collection)
    v = tf.transpose(v)
    # Embed(y); same as V^Ty (, assuming y is a one-hot vector)
    temp = tf.nn.embedding_lookup(v, y)
    # Embed(y) . h
    temp = tf.reduce_sum(temp * global_pooled, axis=1, keep_dims=True)
    return temp


def G_Resblock(name, inputs, nums_out, train_phase, split_z, embed_y, is_up=True):
    """A residual block in BigGAN's generator"""
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = conditional_batchnorm(inputs, train_phase, "bn1", split_z, embed_y)
        inputs = tf.nn.relu(inputs)
        if is_up:
            inputs = up_sampling(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, is_sn=True)
        inputs = conditional_batchnorm(inputs, train_phase, "bn2", split_z, embed_y)
        inputs = tf.nn.relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, is_sn=True)
        # skip connection
        if is_up:
            temp = up_sampling(temp)
        temp = conv("identity", temp, nums_out, 1, 1, is_sn=True)
    return inputs + temp


def D_Resblock(name, inputs, nums_out, train_phase, update_collection=None, is_down=True, use_bn=False):
    """A residual block in BigGAN's discriminator"""
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        if use_bn:
            inputs = conditional_batchnorm(inputs, train_phase, "BN1")
        inputs = tf.nn.relu(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection, is_sn=True)
        if use_bn:
            inputs = conditional_batchnorm(inputs, train_phase, "BN2")
        inputs = tf.nn.relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection, is_sn=True)
        if is_down:
            inputs = down_sampling(inputs)
            # skip connection
            temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=True)
            temp = down_sampling(temp)
        else:
            temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=True)
    return inputs + temp
