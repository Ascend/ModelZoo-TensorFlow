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
# Copyright 2020 Huawei Technologies Co., Ltd
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

""" This module handles the expporting of trained models"""
import os.path
import tensorflow as tf
import model


def export_embeddings(settings, outputdir):
    """ Builds an model using the given settings, loads the last checkpoint and saves only the embedding-variable to a new checkpoint inside outputdir,
        leaving out all the other weights. The new checkpoint is much smaller then the original.
        This new Checkpoint can be used for inference but not to continue training.

        :param ftodtf.settings.FasttextSettings settings: The settings for the model
        :param str outputdir: The directory to store the new checkpoint to.
    """
    m = model.InferenceModel(settings)
    sess = tf.Session(graph=m.graph)
    m.load(settings.log_dir, sess)
    with m.graph.as_default():
        exporter = tf.train.Saver(
            save_relative_paths=True, var_list=m.embeddings, filename="embeddings")
        exporter.save(sess, os.path.join(outputdir, "embeddings"))
