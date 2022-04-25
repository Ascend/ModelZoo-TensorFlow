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
from utils import dataset
import argparse, os


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_url", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--data", type=str, default='UT',choices=['MIT', 'UT', 'MITg', 'UTg'],help="Dataset name")
    parser.add_argument("--test_bz", type=int, default=1024, help="Test batch size")
    parser.add_argument("--obj_pred", type=str, default=None, help="Object prediction from pretrained model")
    parser.add_argument("--bin_path", type=str, default='./data/bin_file/')
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    test_dataloader = dataset.get_dataloader(args.data_url, args.data, 'test', batchsize=args.test_bz,
                                             obj_pred=args.obj_pred)
    input_node = ["Placeholder_2", "test_att_id", "test_obj_id", "Placeholder_6"]
    if not os.path.exists(args.bin_path):
        os.mkdir(args.bin_path)

    for node in input_node:
        if not os.path.exists(args.bin_path + node):
            os.mkdir(args.bin_path + node + "/")

    dset = test_dataloader.dataset
    test_att_id = np.array([dset.attr2idx[attr] for attr, _ in dset.pairs], dtype=np.int32)
    test_obj_id = np.array([dset.obj2idx[obj] for _, obj in dset.pairs], dtype=np.int32)

    count = 0
    for image_ind, batch in enumerate(test_dataloader):
        placeholder_2 = np.array(batch[4])
        placeholder_6 = np.array(batch[-1])

        for i in range(0, len(placeholder_2)):
            placeholder_2[i, :].tofile(args.bin_path + input_node[0] + "/{0:05d}.bin".format(count))
            test_att_id.tofile(args.bin_path + input_node[1] + "/{0:05d}.bin".format(count))
            test_obj_id.tofile(args.bin_path + input_node[2] + "/{0:05d}.bin".format(count))
            placeholder_6[i, :].tofile(args.bin_path + input_node[3] + "/{0:05d}.bin".format(count))
            count += 1


if __name__ == '__main__':
    main()