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

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from data_generator import *
from metrics import dice_coef, dice_loss


def mask_to_3d(mask):
    """
    :param :mask
    :return:mask
    """
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    model_path = "files/resunetplusplus.h5"
    save_path = "result"
    test_path = "new_data/kvasir_segmentation_dataset/test/"

    image_size = 256
    batch_size = 1

    test_image_paths = glob(os.path.join(test_path, "images", "*"))
    test_mask_paths = glob(os.path.join(test_path, "masks", "*"))
    test_image_paths.sort()
    test_mask_paths.sort()

    ## Create result folder
    try:
        os.mkdir(save_path)
    except:
        pass

    ## Model
    with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
        model = load_model(model_path)

    ## Test
    print("Test Result: ")
    test_steps = len(test_image_paths)//batch_size
    test_gen = DataGen(image_size, test_image_paths, test_mask_paths, batch_size=batch_size)
    model.evaluate_generator(test_gen, steps=test_steps, verbose=1)

    ## Generating the result
    for i, path in tqdm(enumerate(test_image_paths), total=len(test_image_paths)):
        image = parse_image(test_image_paths[i], image_size)
        mask = parse_mask(test_mask_paths[i], image_size)

        predict_mask = model.predict(np.expand_dims(image, axis=0))[0]
        predict_mask = (predict_mask > 0.5) * 255.0

        sep_line = np.ones((image_size, 10, 3)) * 255

        mask = mask_to_3d(mask)
        predict_mask = mask_to_3d(predict_mask)

        all_images = [image * 255, sep_line, mask * 255, sep_line, predict_mask]
        cv2.imwrite(f"{save_path}/{i}.png", np.concatenate(all_images, axis=1))

    print("Test image generation complete")
