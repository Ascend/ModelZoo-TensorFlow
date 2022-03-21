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
 
import os
import numpy as np
from evaluation.lfw import LFWTest
from utils.dataset import Dataset
from utils import utils

msameoutdir = './output_bin/2022120_0_50_11_580762'
dataset_path = "./lfw_nooverlap.txt"
protocol_path = "./lfw_pairs.txt"



def main():
    mu = np.array([]).reshape(-1,512)
    sigma_sq = np.array([]).reshape(-1,512)
    print(mu.shape)

    for idx in range(13233):
        output_0 = f"{idx}_output_0.bin"
        output_1 = f"{idx}_output_1.bin"

        a = np.fromfile(msameoutdir + '/' + output_0, dtype='float32')
        a = a.reshape(-1, 512)
        mu = np.vstack([mu, a])

        b = np.fromfile(msameoutdir + '/' + output_1, dtype='float32')
        b = b.reshape(-1, 512)
        sigma_sq = np.vstack([sigma_sq, b])
    feat_pfe = np.concatenate([mu, sigma_sq], axis=1)


    paths = Dataset(dataset_path)['abspath']
    lfwtest = LFWTest(paths)
    lfwtest.init_standard_proto(protocol_path)
    accuracy, threshold = lfwtest.test_standard_proto(mu, utils.pair_euc_score)
    print('Euclidean (cosine) accuracy: %.5f threshold: %.5f' % (accuracy, threshold))
    accuracy, threshold = lfwtest.test_standard_proto(feat_pfe, utils.pair_MLS_score)
    print('MLS accuracy: %.5f threshold: %.5f' % (accuracy, threshold))


if __name__ == "__main__":
    main()