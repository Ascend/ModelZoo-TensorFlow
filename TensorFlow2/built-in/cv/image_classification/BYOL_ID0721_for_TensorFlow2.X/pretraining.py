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
import npu_device as npu
npu.open().as_default()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import tensorflow as tf
import time

from datasets import CIFAR10
from models import ResNet18, ResNet34, ProjectionHead
from losses import byol_loss

encoders = {'resnet18': ResNet18, 'resnet34': ResNet34}

def main(arg):
    '''
    main train
    :param arg:
    :return: loss and weights
    '''
    # Load CIFAR-10 dataset
    data = CIFAR10()
    # Instantiate networks
    f_online = encoders[arg.encoder]()
    g_online = ProjectionHead()
    q_online = ProjectionHead()

    f_target = encoders[arg.encoder]()
    g_target = ProjectionHead()

    # Initialize the weights of the networks
    x = tf.random.normal((256, 32, 32, 3))
    h = f_online(x, training=False)
    print('Initializing online networks...')
    print('Shape of h:', h.shape)
    z = g_online(h, training=False)
    print('Shape of z:', z.shape)
    p = q_online(z, training=False)
    print('Shape of p:', p.shape)

    h = f_target(x, training=False)
    print('Initializing target networks...')
    print('Shape of h:', h.shape)
    z = g_target(h, training=False)
    print('Shape of z:', z.shape)
    
    num_params_f = tf.reduce_sum([tf.reduce_prod(var.shape) for var in f_online.trainable_variables])    
    print('The encoders have {} trainable parameters each.'.format(num_params_f))

    # Define optimizer
    lr = 1e-3 * arg.batch_size / 512
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    print('Using Adam optimizer with learning rate {}.'.format(lr))


    @tf.function
    def train_step_pretraining(x1, x2):  # (bs, 32, 32, 3), (bs, 32, 32, 3)
        #Forward pass
        h_target_1 = f_target(x1, training=True)
        z_target_1 = g_target(h_target_1, training=True)

        h_target_2 = f_target(x2, training=True)
        z_target_2 = g_target(h_target_2, training=True)

        with tf.GradientTape(persistent=True) as tape:
            h_online_1 = f_online(x1, training=True)
            z_online_1 = g_online(h_online_1, training=True)
            p_online_1 = q_online(z_online_1, training=True)
            
            h_online_2 = f_online(x2, training=True)
            z_online_2 = g_online(h_online_2, training=True)
            p_online_2 = q_online(z_online_2, training=True)
            
            p_online = tf.concat([p_online_1, p_online_2], axis=0)
            z_target = tf.concat([z_target_2, z_target_1], axis=0)
            loss = byol_loss(p_online, z_target)

        # Backward pass (update online networks)
        grads = tape.gradient(loss, f_online.trainable_variables)
        opt.apply_gradients(zip(grads, f_online.trainable_variables))
        grads = tape.gradient(loss, g_online.trainable_variables)
        opt.apply_gradients(zip(grads, g_online.trainable_variables))
        grads = tape.gradient(loss, q_online.trainable_variables)
        opt.apply_gradients(zip(grads, q_online.trainable_variables))
        del tape

        return loss

    if 0 == args.perf:
        batches_per_epoch = 20
    else:
        batches_per_epoch = data.num_train_images // arg.batch_size
    log_every = 10  # batches
    save_every = 100  # epochs
    losses = []
    for epoch_id in range(arg.num_epochs):
        data.shuffle_training_data()
        start_time = time.time()
        for batch_id in range(batches_per_epoch):
            x1, x2 = data.get_batch_pretraining(batch_id, arg.batch_size)
            loss = train_step_pretraining(x1, x2)
            losses.append(float(loss))
            # Update target networks (exponential moving average of online networks)
            beta = 0.99

            f_target_weights = f_target.get_weights()
            f_online_weights = f_online.get_weights()
            for i in range(len(f_online_weights)):
                f_target_weights[i] = beta * f_target_weights[i] + (1 - beta) * f_online_weights[i]
            f_target.set_weights(f_target_weights)

            g_target_weights = g_target.get_weights()
            g_online_weights = g_online.get_weights()
            for i in range(len(g_online_weights)):
                g_target_weights[i] = beta * g_target_weights[i] + (1 - beta) * g_online_weights[i]
            g_target.set_weights(g_target_weights)

            if (batch_id + 1) % log_every == 0:
                end_time = time.time()
                print('[Epoch {}/{} Batch {}/{}] Loss={:.5f} Time={}.'.format(epoch_id + 1, arg.num_epochs, batch_id + 1,
                                                                      batches_per_epoch, loss, end_time-start_time))
                start_time = time.time()
        if (epoch_id + 1) % save_every == 0:
            f_online.save_weights('f_online_{}.h5'.format(epoch_id + 1))
            print('Weights of f saved.')

    np.savetxt('losses.txt', tf.stack(losses).numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder', type=str, default='resnet18', choices=['resnet18', 'resnet34'], help='Encoder architecture')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--perf', type=int, default=1, help='perf')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for pretraining')
    
    args = parser.parse_args()
    main(args)
