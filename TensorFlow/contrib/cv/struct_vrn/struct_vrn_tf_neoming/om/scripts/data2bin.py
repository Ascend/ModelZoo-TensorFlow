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

# Lint as: python3
r"""Minimal example for training a video_structure model.

See README.md for installation instructions. To run on GPU device 0:

CUDA_VISIBLE_DEVICES=0 python -m video_structure.train
"""

import os
import datasets
import hyperparameters
import tensorflow as tf
import numpy as np

def eval():
  cfg = hyperparameters.get_config()

  test_dataset, _ = datasets.get_sequence_dataset(
      data_dir=os.path.join(cfg.data_dir, cfg.test_dir),
      batch_size=cfg.batch_size,
      num_timesteps=cfg.observed_steps + cfg.predicted_steps)
  iterator = test_dataset.make_one_shot_iterator()
  for i in range(100):
      data = iterator.get_next()
      data = data['image']
      data = tf.Session().run(data)
      data.tofile("om/bindata/batch_{}.bin".format(i))



if __name__ == '__main__':
  
  eval()
