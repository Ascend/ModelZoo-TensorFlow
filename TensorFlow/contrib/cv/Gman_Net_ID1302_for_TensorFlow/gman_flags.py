"""
flags
"""
# coding=utf-8
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

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# Dehazenet actually doesn't require very high float data accuracy,
# so fp16 is normally set False
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_string('train_url', 'obs://imagenet2012-lp/GMan_log/', """the path of train log in obs""")
# tf.app.flags.DEFINE_string('data_url', 'obs://imagenet2012-lp/coltran_re/imagenet2012/', """the path of data in obs""")
tf.app.flags.DEFINE_string('data_url', 'obs://imagenet2012-lp/Gman_re/Train_record/train.tfrecords',
                           """the path of data in obs""")
tf.app.flags.DEFINE_string('data_path', '/home/ma-user/modelarts/inputs/data_url_0/train.tfrecords',
                           """the path of data in obs""")
tf.app.flags.DEFINE_string('data_test_path', '/home/ma-user/modelarts/inputs/data_url_0/test.tfrecords',
                           """the path of data in obs""")
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('input_image_height', 224,
                            """Input image height.""")
tf.app.flags.DEFINE_integer('input_image_width', 224,
                            """Input image width.""")
tf.app.flags.DEFINE_integer('original_height', 100000,
                            """Input image original height.""")
tf.app.flags.DEFINE_integer('original_width', 100000,
                            """Input image original width.""")
tf.app.flags.DEFINE_string('haze_train_images_dir', '/HazeImages/TrainImages',
                           """Path to the hazed train images directory.""")
tf.app.flags.DEFINE_string('clear_train_images_dir', './ClearImages/TrainImages',
                           """Path to the clear train images directory.""")
tf.app.flags.DEFINE_string('tfrecord_format', 'gman-%d.tfrecords',
                           """Format of tf-records, file name must end with -index_number.""")
tf.app.flags.DEFINE_string('tfrecord_json', './TFRecord/tfrecords.json',
                           """Json file to save the status of tfrecords.""")
tf.app.flags.DEFINE_string('tfrecord_path', './TFRecord',
                           """Path to save tfrecords.""")
tf.app.flags.DEFINE_boolean('tfrecord_rewrite', False,
                            """Whether to delete and rewrite the TFRecord.""")
tf.app.flags.DEFINE_string('PerceNet_dir', './PerceNetModel/vgg16.npy',
                           """Path to save the PerceNet Model""")
tf.app.flags.DEFINE_boolean('train_restore', False,
                            """Whether to restore the trained model.""")
tf.app.flags.DEFINE_string('train_json_path', '/trainFlowControl.json',
                           """Path to save training status json file.""")
tf.app.flags.DEFINE_integer('max_epoch', 500,
                            """Max epoch number for training.""")
tf.app.flags.DEFINE_string('train_learning_rate', '/trainLearningRate.json',
                           """Path to save training learning rate json file.""")

# Some systematic parameters
tf.app.flags.DEFINE_string('train_dir', '/cache/saveModels/DeHazeNetModel',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 19990000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Variables for evaluation
tf.app.flags.DEFINE_string('checkpoint_dir_obs',
                           'obs://imagenet2012-lp/GMan_log/MA-new-GMan_modelarts-12-09-14-21/output/DeHazeNetModel/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('eval_dir', './DeHazeNet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/cache/saveModels/DeHazeNetModel',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('haze_test_images_dir', '/home/ma-user/modelarts/inputs/data_url_0/HazedImages/TestImages/',
                           """Path to the hazed test images directory.""")
tf.app.flags.DEFINE_string('clear_test_images_dir', '/home/ma-user/modelarts/inputs/data_url_0/CleanImages/TestImages/',
                           """Path to the clear train images directory.""")
tf.app.flags.DEFINE_string('clear_result_images_dir', '/cache/saveModels/ClearResultImages/',
                           """Path to the dehazed test images directory.""")
tf.app.flags.DEFINE_string('tfrecord_eval_path', './TFRecord/eval.tfrecords',
                           """Path to save the test TFRecord of the images""")
tf.app.flags.DEFINE_boolean('tfrecord_eval_rewrite', False,
                            """Whether to delete and rewrite the TFRecord.""")
tf.app.flags.DEFINE_string('save_image_type', 'jpg',
                           """In which format to save image.""")

tf.app.flags.DEFINE_boolean('eval_log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('eval_max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('eval_num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('eval_input_image_height', 128,
                            """Input image height.""")
tf.app.flags.DEFINE_integer('eval_input_image_width', 128,
                            """Input image width.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 10,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1000000000,
                            """Number of test examples to run""")
tf.app.flags.DEFINE_boolean('eval_only_haze', False,
                            """Whether to load clear images.""")
