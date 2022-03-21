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


import os
import argparse
import glob
import numpy as np
from acl_utils import compute_errors, load_test_data, image_resize, scale_up, DepthNorm


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bs', default=1,
                        help="""batchsize""")
    parser.add_argument('--bin_dir', default='./bin/image',
                        help="""the bin data path""")
    parser.add_argument('--bin_flip_dir', default='./bin/image_flip',
                        help="""the bin data path""")
    parser.add_argument('--nyu_dir', default='./dataset/nyu_test.zip',
                        help="""the nyu data path""")
    parser.add_argument('--minDepth', type=float, default=10.0, help='Minimum of input depths')
    parser.add_argument('--maxDepth', type=float, default=1000.0, help='Maximum of input depths')

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


class Classifier(object):

    def __init__(self):

        args = parse_args()
        self.batch_size = int(args.bs)
        self.nyu_path = args.nyu_dir
        self.maxDepth = args.maxDepth
        self.minDepth = args.minDepth
        self.image_dir = args.bin_dir
        self.image_flip_dir = args.bin_flip_dir

    def load_bin(self, bin_path):

        x = np.fromfile(bin_path, dtype=np.float32)
        x = np.reshape(x, (self.batch_size, 240, 320, 1))
        x = np.clip(DepthNorm(x, maxDepth=self.maxDepth), self.minDepth, self.maxDepth) / self.maxDepth

        return x

    def read_file(self, file_path):

        bin_files = os.listdir(file_path)
        loaded_bin_dic = {}
        for bin_file in bin_files:
            idx = bin_file.split("_")[3]
            loaded_bin_dic.update({str(idx): bin_file})

        return loaded_bin_dic
    def read_file_flip(self, file_path):

        bin_files = os.listdir(file_path)
        loaded_bin_dic = {}
        for bin_file in bin_files:
            idx = bin_file.split("_")[4]
            loaded_bin_dic.update({str(idx): bin_file})

        return loaded_bin_dic

    def load_nyu(self, nyu_path):

        nyu_test_data = load_test_data(nyu_path)

        return nyu_test_data

    def compute_process(self, pre_path, pred_flip_path, crop, depth):
        predictions = []
        testSetDepths = []
        bs = self.batch_size
        for i in range(len(pre_path)):

            # load data
            pre = self.load_bin(os.path.join(self.image_dir, pre_path[str(i)]))
            pred_flip = self.load_bin(os.path.join(self.image_flip_dir, pred_flip_path[str(i)]))

            # Compute results
            true_y = depth[i * bs:(i + 1) * bs, :, :]

            pred_y = scale_up(2, pre[:, :, :, 0]) * 10.0

            # Test time augmentation: mirror image estimate
            pred_y_flip = scale_up(2, pred_flip[:, :, :, 0]) * 10.0

            # Crop based on Eigen et al. crop
            true_y = true_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
            pred_y = pred_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
            pred_y_flip = pred_y_flip[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]

            # Compute errors per image in batch
            for j in range(len(true_y)):
                predictions.append((0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j])))
                testSetDepths.append(true_y[j])

        predictions = np.stack(predictions, axis=0)
        testSetDepths = np.stack(testSetDepths, axis=0)

        return predictions, testSetDepths


def main():
    args = parse_args()
    # data load and preprocess
    classifier = Classifier()

    print("########NOW Start Load Data!!!#########")

    # load bin data
    image_path = classifier.read_file(args.bin_dir)
    print("bin images data loaded")
    image_flip_path = classifier.read_file_flip(args.bin_flip_dir)
    print("bin flip images data loaded")

    # load nyu test data
    nyu_data = classifier.load_nyu(args.nyu_dir)
    crop = nyu_data["crop"]
    depth = nyu_data["depth"]

    print("########NOW Start Data Pretreatment!!!#########")

    predictions, testSetDepths = classifier.compute_process(image_path, image_flip_path, crop, depth)

    print("########NOW Start Compute Accuary!!!#########")
    # compute accuracy
    e = compute_errors(predictions, testSetDepths)

    print('+----------------------------------------+')
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4], e[5]))
    print('+----------------------------------------+')


if __name__ == '__main__':
    main()
