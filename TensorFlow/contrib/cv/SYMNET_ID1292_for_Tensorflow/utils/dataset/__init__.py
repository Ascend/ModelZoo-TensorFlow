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
from .. import config as cfg
from utils.dataset import CZSL_dataset
from torch.utils.data import DataLoader
import numpy as np
import os.path as osp
import os
cwd = os.getcwd()
def get_dataloader(train_url, dataset_name, phase, feature_file="features.t7", batchsize=1, num_workers=1, shuffle=None, **kwargs):
    print("url:",train_url)
    print("url:", cfg.CZSL_DS_ROOT[dataset_name])
    dt_path = osp.join(train_url, cfg.CZSL_DS_ROOT[dataset_name])
    # /home/ma-user/modelarts/inputs/data_url_0/ut-zap50k-original
    print("dt_path",dt_path)

    print("Current working directory: {0}".format(cwd))

    dataset = CZSL_dataset.CompositionDatasetActivations(
        train_url = train_url,
        name = dataset_name,
        root = dt_path,     #data/mit-states-original/features.t7
        phase = phase,
        feat_file = feature_file,
        **kwargs)
    

    if shuffle is None:
        shuffle = (phase=='train')
    
    return DataLoader(dataset, batchsize, shuffle, num_workers=num_workers,
        collate_fn = lambda data: [np.stack(d, axis=0) for d in zip(*data)]
    )


    

