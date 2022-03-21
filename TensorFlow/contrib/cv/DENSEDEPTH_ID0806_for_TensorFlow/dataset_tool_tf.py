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
import sys
import argparse
import tensorflow as tf
import random
from utils import extract_zip


def shape_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_nyu_data(batch_size, nyu_data_zipfile='nyu_data.zip'):
    data = extract_zip(nyu_data_zipfile)

    nyu2_train = list(
        (row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_test = list(
        (row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)

    # Helpful for testing...
    if False:
        nyu2_train = nyu2_train[:5000]
        nyu2_test = nyu2_test[:20]

    return data, nyu2_train, nyu2_test, shape_rgb, shape_depth


examples = '''examples:

  python %(prog)s --input-dir=./kodak --out=imagenet_val_raw.tfrecords
'''


def main():
    parser = argparse.ArgumentParser(
        description='Convert a set of image files into a TensorFlow tfrecords training set.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input_dir", default=r'./dataset/nyu_data.zip', help="Directory containing ImageNet images")
    parser.add_argument("--out", default=r'./dataset/nyu_data.tfrecords', help="Filename of the output tfrecords file")
    args = parser.parse_args()

    if args.input_dir is None:
        print('Must specify input file directory with --input-dir')
        sys.exit(1)
    if args.out is None:
        print('Must specify output filename with --out')
        sys.exit(1)

    batch_size = 1

    print('Loading image list from %s' % args.input_dir)
    data, nyu2_train, nyu2_test, shape_rgb, shape_depth = get_nyu_data(batch_size, args.input_dir)
    random.shuffle(nyu2_train)
    N = len(nyu2_train)

    # ----------------------------------------------------------
    outdir = os.path.dirname(args.out)
    os.makedirs(outdir, exist_ok=True)

    writer = tf.python_io.TFRecordWriter(args.out)

    for (idx, imgname) in enumerate(nyu2_train):
        print(idx, imgname)
        # Augmentation of RGB images
        index = min((idx), N - 1)

        sample = nyu2_train[index]
        #######################################################################################
        # 其他设置（自定义）
        sample = list(s.strip() for s in sample)
        #######################################################################################

        x = data[sample[0]]
        y = data[sample[1]]

        feature = {
            'data_x': bytes_feature(x),
            'data_y': bytes_feature(y),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    print('Dataset done.')
    writer.close()


if __name__ == "__main__":
    main()
