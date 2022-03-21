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
# ============================================================================
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import json


"""
************ATTENTION! *****************
the generated image vectors of all clothings are queit big, so please set the suitable directory for saving them.
"""
image = tf.keras.preprocessing.image
preprocess = tf.keras.applications.inception_v3.preprocess_input
myinception = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    pooling='max'
)
file_path = "./images/"
outfit_list = os.listdir(file_path)

for idx, outfit_id in enumerate(outfit_list):
    file_path_outfit = file_path + outfit_id + '/'
    item_list = os.listdir(file_path_outfit)
    print (idx, len(outfit_list))
    for item_id in item_list:
        file_path_outfit_item = file_path_outfit + item_id + '/'

        img = Image.open(file_path_outfit_item)
        img = img.resize((229, 229))
        mat = image.img_to_array(img)
        mat = np.expand_dims(mat, axis=0)
        aa = preprocess(mat)

        itemvector = myinception.predict(aa)

        vector_name = outfit_id + '_' + item_id + '.json'
        with open('./polyvore_image_vectors/' + vector_name, 'w') as f:
            f.write(json.dumps(list(itemvector[0])))


