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



import numpy as np
import pandas as pd
from kutils import applications as apps
from kutils import image_utils as img


def data_process(images_path, dataset, output):
    ids = pd.read_csv(dataset)
    pre = apps.process_input[apps.InceptionResNetV2]
    images_path = images_path + ids[ids.set == 'test'].iloc[0].image_name
    img_name = ids[ids.set == 'test'].iloc[0].image_name
    print(img_name)
    # load, pre-process it, and pass it to the model
    I = pre(img.read_image(images_path))
    I = np.expand_dims(I, 0)
    print(I)
    I.tofile(output + '2' + ".bin")


if __name__ == '__main__':
    data_path = 'E:/Program/ava-mlsp/'
    dataset = data_path + 'metadata/AVA_data_official_test.csv'
    images_path = data_path + 'images/'
    output = 'E:/Program/images/'
    data_process(images_path, dataset, output)
