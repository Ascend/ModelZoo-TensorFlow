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

from generators.pascal import PascalVocGenerator
import cv2
import tqdm
import os
import numpy as np
import sys
import os.path as osp


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
generator = PascalVocGenerator(
    'VOC2007',
    'test',
    shuffle_groups=False,
    skip_truncated=False,
    skip_difficult=True,
)
model_path = './pascal_81_1.5415_3.0741_0.6860_0.7057_0.7209_0.7290.h5'
num_classes = generator.num_classes()
classes = list(generator.classes.keys())
flip_test = True
nms = True
keep_resolution = False
score_threshold = 0.1
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

for i in tqdm.tqdm(range(4952)):
    image = generator.load_image(i)
    src_image = image.copy()
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    s = max(image.shape[0], image.shape[1]) * 1.0

    tgt_w = generator.input_size
    tgt_h = generator.input_size
    image = generator.preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)
    
    if flip_test:
        flipped_image = image[:, ::-1]
        inputs = np.stack([image, flipped_image], axis=0)
    else:
        inputs = np.expand_dims(image, axis=0)
    
    input_path = sys.argv[1]
    inputs.tofile(os.path.join(input_path, generator.image_names[i]+"bin"))
