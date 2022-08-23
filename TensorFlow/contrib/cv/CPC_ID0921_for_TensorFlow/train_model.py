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
'''
This module describes the contrastive predictive coding model from DeepMind:

Oord, Aaron van den, Yazhe Li, and Oriol Vinyals.
"Representation Learning with Contrastive Predictive Coding."
arXiv pre# print arXiv:1807.03748 (2018).
'''
import os

import argparse
from data_utils import SortedNumberGenerator
from os.path import join, basename, dirname, exists

from tensorflow.python import keras
from tensorflow.python.keras import backend as K

import tensorflow as tf
from npu_bridge.npu_init import *

sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)


def network_encoder(x, code_size):
    ''' Define the network mapping images to embeddings '''

    x = keras.layers.Conv2D(filters=64,
                            kernel_size=3,
                            strides=2,
                            activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(filters=64,
                            kernel_size=3,
                            strides=2,
                            activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(filters=64,
                            kernel_size=3,
                            strides=2,
                            activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(filters=64,
                            kernel_size=3,
                            strides=2,
                            activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size,
                           activation='linear',
                           name='encoder_embedding')(x)

    return x


def network_autoregressive(x):
    ''' Define the network that integrates information along the sequence '''

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(units=256, return_sequences=False,
                         name='ar_context')(x)

    return x


def network_prediction(context, code_size, predict_terms):
    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(
            keras.layers.Dense(units=code_size,
                               activation="linear",
                               name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(
            outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.layers.Layer):
    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(
            dot_product, axis=-1,
            keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def network_cpc(image_shape, terms, predict_terms, code_size, learning_rate):
    ''' Define the CPC network combining encoder and autoregressive model '''

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)

    # Define encoder model
    encoder_input = keras.layers.Input(image_shape)
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input,
                                       encoder_output,
                                       name='encoder')
    encoder_model.summary()

    # Define rest of model
    x_input = keras.layers.Input(
        (terms, image_shape[0], image_shape[1], image_shape[2]))
    x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input)
    context = network_autoregressive(x_encoded)
    preds = network_prediction(context, code_size, predict_terms)

    y_input = keras.layers.Input(
        (predict_terms, image_shape[0], image_shape[1], image_shape[2]))
    y_encoded = keras.layers.TimeDistributed(encoder_model)(y_input)

    # Loss
    dot_product_probs = CPCLayer()([preds, y_encoded])

    # Model
    cpc_model = keras.models.Model(inputs=[x_input, y_input],
                                   outputs=dot_product_probs)

    # Compile model
    cpc_model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])
    cpc_model.summary()

    return cpc_model


def train_model(epochs,
                batch_size,
                output_dir,
                code_size,
                lr=1e-4,
                terms=4,
                predict_terms=4,
                image_size=28,
                color=False):
    # Prepare data
    train_data = SortedNumberGenerator(batch_size=batch_size,
                                       subset='train',
                                       terms=terms,
                                       positive_samples=batch_size // 2,
                                       predict_terms=predict_terms,
                                       image_size=image_size,
                                       color=color,
                                       rescale=True)

    validation_data = SortedNumberGenerator(batch_size=batch_size,
                                            subset='valid',
                                            terms=terms,
                                            positive_samples=batch_size // 2,
                                            predict_terms=predict_terms,
                                            image_size=image_size,
                                            color=color,
                                            rescale=True)

    model = network_cpc(image_shape=(image_size, image_size, 3),
                        terms=terms,
                        predict_terms=predict_terms,
                        code_size=code_size,
                        learning_rate=lr)

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                          factor=1 / 3,
                                          patience=2,
                                          min_lr=1e-4)
    ]

    # Trains the model
    model.fit_generator(generator=train_data,
                        steps_per_epoch=len(train_data),
                        validation_data=validation_data,
                        validation_steps=len(validation_data),
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks)

    # mkdir
    os.makedirs(output_dir)

    # Saves the model
    # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
    model.save(join(output_dir, 'cpc.h5'))

    # Saves the encoder alone
    encoder = model.layers[1].layer
    encoder.save(join(output_dir, 'encoder.h5'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()
    train_model(epochs=args.epoch,
                batch_size=args.batch_size,
                output_dir='output',
                code_size=128,
                lr=args.lr,
                terms=4,
                predict_terms=4,
                image_size=64,
                color=True)
    sess.close()
