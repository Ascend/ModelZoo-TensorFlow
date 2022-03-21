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


import argparse

from classificationDataTool import ClassificationImageData


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='from which to generate TFRecord, folders or mxrec',
                        default='mxrec')
    parser.add_argument('--image_size', type=int, help='image size', default=112)
    parser.add_argument('--read_dir', type=str, help='directory to read data',
                        default='')
    parser.add_argument('--save_path', type=str, help='path to save TFRecord file',
                        default='')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cid = ClassificationImageData(img_size=args.image_size)
    if args.mode == 'folders':
        cid.write_tfrecord_from_folders(args.read_dir, args.save_path)
    elif args.mode == 'mxrec':
        cid.write_tfrecord_from_mxrec(args.read_dir, args.save_path)
    else:
        raise ('ERROR: wrong mode (only folders and mxrec are supported)')
