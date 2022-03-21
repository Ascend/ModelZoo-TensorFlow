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
from urllib import request
import gzip
import pickle
import os

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist(datadir = "./datasets/MNIST"):
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    else:
        return;
    
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        path = os.path.join(datadir, name[1])
        request.urlretrieve(base_url+name[1], path)
    print("Download complete.")

def save_mnist(datadir = "./datasets/MNIST"):
    mnist = {}
    for name in filename[:2]:
        path = os.path.join(datadir, name[1])
        with gzip.open(path, 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        path = os.path.join(datadir, name[1])
        with gzip.open(path, 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    path = os.path.join(datadir, "mnist.pkl")
    with open(path, 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init(datadir = "./datasets/MNIST"):
    download_mnist(datadir)
    save_mnist(datadir)


def load(datadir = "./datasets/MNIST"):
    path = os.path.join(datadir,"mnist.pkl")
    if not os.path.exists(path):
        init(datadir)
    with open(path,'rb') as f:
        mnist = pickle.load(f)

    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

import cv2
def imgs_resize(imgs):
    imgs_32 = []
    for i in imgs:
        tmp = cv2.resize(i.reshape(28,28), (32,32))
        imgs_32.append(tmp)
    #     break
    imgs_32 = np.array(imgs_32)
    return imgs_32
if __name__ == '__main__':
    init()
