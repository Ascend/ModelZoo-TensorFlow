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


from __future__ import absolute_import
from npu_bridge.npu_init import *
import tensorflow as tf
from configs import configs
from squeezenext_model import Model
from scipy import misc
import numpy as np
import scipy
import argparse
from datasets.build_imagenet_data import _build_synset_lookup
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('image_path', type=str, help='Location of eval jpeg image')
parser.add_argument('--model_dir', type=str, required=True, help='Location of model_dir')
parser.add_argument('--configuration', type=str, default='v_1_0_SqNxt_23_mod', help='Name of model config file')
parser.add_argument('--imagenet_metadata_file', type=str, default='./datasets/imagenet_metadata.txt', help='Path to metadata file')
parser.add_argument('--labels_file', type=str, default='./datasets/imagenet_lsvrc_2015_synsets.txt', help='Path to labels file')
args = parser.parse_args()

def lookup_human_readable(res, synset, lookup_table):
    return lookup_table[synset[res]]

def main(argv):
    '\n    Main function to start training\n    :param argv:\n        not used\n    :return:\n        None\n    '
    del argv
    config = configs[args.configuration]
    config['model_dir'] = args.model_dir
    config['output_train_images'] = False
    config['total_steps'] = 1
    config['fine_tune_ckpt'] = None
    model = Model(config, 1)
    classifier = tf.estimator.Estimator(model_dir=args.model_dir, model_fn=model.model_fn, params=config, config=npu_run_config_init())
    image = misc.imread(args.image_path)
    resized = scipy.misc.imresize(image, (256, 256, 3))
    crop_min = abs(((config['image_size'] / 2) - (config['image_size'] / 2)))
    crop_max = (crop_min + config['image_size'])
    image = resized[crop_min:crop_max, crop_min:crop_max, :]
    mean_sub = (image.astype(np.float32) - np.array([123, 117, 104]).astype(np.float32))
    image = np.expand_dims(np.array(mean_sub), 0)
    my_input_fn = tf.estimator.inputs.numpy_input_fn(x={'image': image}, shuffle=False, batch_size=1)
    lookup_table = _build_synset_lookup(args.imagenet_metadata_file)
    challenge_synsets = [l.strip() for l in tf.gfile.FastGFile(args.labels_file, 'r').readlines()]
    predictions = classifier.predict(input_fn=my_input_fn)
    for result in predictions:
        print('top 5: \n 1: {} \n 2: {} \n 3: {} \n 4: {} \n 5: {} \n'.format(lookup_human_readable(result['top_5'][0], challenge_synsets, lookup_table), lookup_human_readable(result['top_5'][1], challenge_synsets, lookup_table), lookup_human_readable(result['top_5'][2], challenge_synsets, lookup_table), lookup_human_readable(result['top_5'][3], challenge_synsets, lookup_table), lookup_human_readable(result['top_5'][4], challenge_synsets, lookup_table)))
if (__name__ == '__main__'):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
