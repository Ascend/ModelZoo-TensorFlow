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
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='./offline_infer/Bin/gt')
parser.add_argument('--output_path', type=str, default='./offline_infer/Bin/outputs_pb', 
                    help='output path.')
args = parser.parse_args()

if __name__ == '__main__':
    gt = args.gt_path
    prediction = args.output_path

    imgs = os.listdir(gt)
    preds = os.listdir(prediction)
    imgs = sorted(imgs)
    preds = sorted(preds)

    res = []

    for img, pred in zip(imgs, preds):
        label = os.path.join(gt, img)
        out = os.path.join(prediction, pred)

        label = np.fromfile(label, dtype=np.float32)
        out = np.fromfile(out, dtype=np.float32)

        label = label.reshape(448, 1024, 2)
        out = out.reshape(448, 1024, 2)
        label = label[:436, :, :]
        out = out[:436, :, :]
    
        res.append(np.mean(np.linalg.norm(label - out, axis=-1)))

    print('Average EPE = ', np.mean(res))

