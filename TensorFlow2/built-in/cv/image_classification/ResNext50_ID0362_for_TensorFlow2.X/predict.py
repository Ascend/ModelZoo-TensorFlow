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
#

import tensorflow as tf
from configuration import save_model_dir, test_image_dir
from prepare_data import load_and_preprocess_image
from train import get_model


def get_single_picture_prediction(model, picture_dir):
    image_tensor = load_and_preprocess_image(tf.io.read_file(filename=picture_dir), data_augmentation=False)
    image = tf.expand_dims(image_tensor, axis=0)
    prediction = model(image, training=False)
    pred_class = tf.math.argmax(prediction, axis=-1)
    return pred_class


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # load the model
    model = get_model()
    model.load_weights(filepath=save_model_dir+"model")

    pred_class = get_single_picture_prediction(model, test_image_dir)
    print(pred_class)