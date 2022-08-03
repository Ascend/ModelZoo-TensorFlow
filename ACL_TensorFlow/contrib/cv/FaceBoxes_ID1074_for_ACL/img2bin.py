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
import json
from PIL import Image, ImageDraw
import os
import cv2
import pandas as pd
from tqdm import tqdm
import shutil
import math
import io
import os
import tensorflow as tf
import PIL.Image
import numpy as np



import random


image_dir=RESULT_DIR+'images/'
annotations_dir=RESULT_DIR+'annotations/'
output_dir=RESULT_DIR+'output2/'



print('Reading images from:', image_dir)
print('Reading annotations from:', annotations_dir, '\n')

examples_list = os.listdir(annotations_dir)
num_examples = len(examples_list)
print('Number of images:', num_examples)


# output_dir = ARGS.output
shutil.rmtree(output_dir, ignore_errors=True)
os.mkdir(output_dir)


for example in tqdm(examples_list):

    path = os.path.join(annotations_dir, example)
    annotation = json.load(open(path))
    # tf_example = dict_to_tf_example(annotation, image_dir)
    image_name = annotation['filename']
    assert image_name.endswith('.jpg') or image_name.endswith('.jpeg')

    image_path = os.path.join(image_dir, image_name)
    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()

    # check image format
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG!')

    image_=np.asarray(image)
    image__=cv2.resize(image_,(1024, 1024), interpolation=cv2.INTER_CUBIC)
    image__.tofile(output_dir+"{}.bin".format(image_name))






print('Result is here:', RESULT_DIR)