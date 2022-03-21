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


from tqdm import tqdm
import os
import sys
import numpy as np
from absl import app, flags
from absl.flags import FLAGS

sys.path.append('./')
from core.dataset import Dataset, DatasetFetcher


flags.DEFINE_string('model', 'yolov5', 'yolov3,yolov4,yolov5')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_integer('batchsize', 1, 'total batchsize across all gpus')
flags.DEFINE_string('data_path', './data/dataset/val2017.txt', 'path of annotation file')
flags.DEFINE_boolean('mosaic', True, 'activate mosaic data augmentation')


def data_preprocess(testset, file_path):
    if os.path.isdir(file_path):
        os.system('rm -rf ' + file_path)
    os.makedirs(file_path)

    pbar = tqdm(testset)
    iterdata = iter(pbar)
    fetcher = DatasetFetcher(testset)
    i = 1
    while True:
        try:
            annotations = next(iterdata)
        except StopIteration:
            break
        if annotations is None:
            break
        test_data, _, batch_image_id, scale, dw, dh = fetcher.process_annotations(annotations)
        np.array(test_data, dtype=np.float32).tofile(file_path + '/' + str(i.__str__()).zfill(6) + '.bin')
        i += 1


def main(_argv):
    file_path = './offline_inference/input_bins'
    testset = Dataset(FLAGS, is_training=False)
    data_preprocess(testset, file_path)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
