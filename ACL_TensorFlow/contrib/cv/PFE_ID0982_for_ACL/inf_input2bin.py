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
from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from network import Network
import time
import sys
model_dir = "/home/ma-user/modelarts/user-job-dir/code/log/sphere64_casia_am_PFE/20211215-121426"
dataset_path = "data/lfw_nooverlap.txt"
def main():
    network = Network()
    network.load_model(model_dir)
    paths = Dataset(dataset_path)['abspath']
    images = preprocess(paths, network.config, False)
    print(images.shape)
    batch_size = 1
    num_images = len(images)
    for start_idx in range(0, num_images, batch_size):
        images_batch = images[start_idx]
        images_batch.tofile(f"./input_bin/{start_idx}.bin")


if __name__ == "__main__":
    main()

