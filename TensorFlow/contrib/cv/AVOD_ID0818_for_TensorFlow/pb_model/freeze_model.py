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


import argparse
import os
import sys

import tensorflow as tf

sys.path.append("..")

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.avod_model import AvodModel
from avod.core.models.rpn_model import RpnModel

from avod.core import trainer_utils

import time
import os
import numpy as np
from multiprocessing import Process

from tensorflow.python.tools import freeze_graph


class Freezer:

    def __init__(self,
                 model,
                 eval_config,
                 ckpt_index):

        # Get model configurations
        self.model = model
        self.eval_config = eval_config

        self.model_config = model.model_config
        self.model_name = self.model_config.model_name
        self.full_model = isinstance(self.model, AvodModel)

        self.paths_config = self.model_config.paths_config
        self.checkpoint_dir = self.paths_config.checkpoint_dir
        
        self.ckpt_index = ckpt_index

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        self._sess = tf.Session()

        # The model should return a dictionary of predictions
        self._prediction_dict = self.model.build()

        self.output_name_list = [tensor.name.split(':')[0] for tensor in self._prediction_dict.values()]

        self.outputs_string = ','.join([tensor.name for tensor in self._prediction_dict.values()])

        self._saver = tf.train.Saver()



    def run_latest_checkpoints(self):

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        if self.ckpt_index == -1:
            checkpoint_to_restore = tf.train.latest_checkpoint(self.checkpoint_dir)
        elif self.ckpt_index >=0:
            trainer_utils.load_checkpoints(self.checkpoint_dir, self._saver)
            checkpoint_to_restore = self._saver.last_checkpoints[self.ckpt_index]

        print('** Checkpoint Path to freeze: {}'.format(checkpoint_to_restore))

        self._saver.restore(self._sess, checkpoint_to_restore)

        tmp_g = self._sess.graph.as_graph_def()

        frozen_graph = tf.graph_util.convert_variables_to_constants(self._sess, 
            tmp_g, self.output_name_list)
        
        out_graph_path = "./pb_model/frozen_model_ckpt_{}.pb".format(self.ckpt_index)

        with tf.io.gfile.GFile(out_graph_path, "wb") as f:
            f.write(frozen_graph.SerializeToString())


def freeze(model_config, eval_config, dataset_config, ckpt_index):

    dataset_config = config_builder.proto_to_obj(dataset_config)
    dataset_config.data_split = 'val'
    dataset_config.has_labels = False
    dataset_config.aug_list = []

    # Build the dataset object
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)


    eval_config = config_builder.proto_to_obj(eval_config)

    # Setup the model
    model_name = model_config.model_name
    # Overwrite repeated field
    model_config = config_builder.proto_to_obj(model_config)
    # Switch path drop off during evaluation
    model_config.path_drop_probabilities = [1.0, 1.0]

    with tf.Graph().as_default():
        model = AvodModel(model_config,
                            train_val_test='test',
                            dataset=dataset)

        model_freezer = Freezer(model, eval_config,ckpt_index)
        model_freezer.run_latest_checkpoints()


def main(_):
    parser = argparse.ArgumentParser()

    # Example usage
    # --checkpoint_name='avod_cars_example'
    # Optional arg:
    # --device=0

    parser.add_argument('--checkpoint_name',
                        type=str,
                        dest='checkpoint_name',
                        required=True,
                        help='Checkpoint name must be specified as a str\
                        and must match the experiment config file name.')

    parser.add_argument(
        '--ckpt_index',
        type=int,
        dest='ckpt_index',
        required=False,
        default=-1,
        help='Checkpoint indices must be an \
        integer in between -> 0 10 20 etc\
        default value -1 : latest checkpoint')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default='0',
                        help='CUDA device id')

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    experiment_config = args.checkpoint_name + '.config'

    # Read the config from the experiment folder
    experiment_config_path = avod.root_dir() + '/data/outputs/' +\
        args.checkpoint_name + '/' + experiment_config

    model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            experiment_config_path, is_training=False)

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    freeze(model_config, eval_config, dataset_config, args.ckpt_index)


if __name__ == '__main__':
    tf.app.run()
