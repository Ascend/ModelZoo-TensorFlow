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
''' This module evaluates the performance of a trained CPC encoder '''

from data_utils import MnistGenerator
from os.path import join, basename, dirname, exists
import keras
import moxing as mox


def build_model(encoder_path, image_shape, learning_rate):

    # Read the encoder
    encoder = keras.models.load_model(encoder_path)

    # Freeze weights
    encoder.trainable = False
    for layer in encoder.layers:
        layer.trainable = False

    # Define the classifier
    x_input = keras.layers.Input(image_shape)
    x = encoder(x_input)
    x = keras.layers.Dense(units=128, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=10, activation='softmax')(x)

    # Model
    model = keras.models.Model(inputs=x_input, outputs=x)

    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model.summary()

    return model


def benchmark_model(encoder_path,
                    epochs,
                    batch_size,
                    output_dir,
                    lr=1e-4,
                    image_size=28,
                    color=False):

    # Prepare data
    train_data = MnistGenerator(batch_size,
                                subset='train',
                                image_size=image_size,
                                color=color,
                                rescale=True)

    validation_data = MnistGenerator(batch_size,
                                     subset='valid',
                                     image_size=image_size,
                                     color=color,
                                     rescale=True)

    # Prepares the model
    model = build_model(encoder_path,
                        image_shape=(image_size, image_size, 3),
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

    # Saves the model
    model.save(join(output_dir, 'supervised.h5'))


if __name__ == "__main__":

    benchmark_model(encoder_path='output/encoder.h5',
                    epochs=15,
                    batch_size=64,
                    output_dir='output',
                    lr=1e-3,
                    image_size=64,
                    color=True)
