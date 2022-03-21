"""
lr
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

import os
import json
# import tensorflow as tf

import gman_constant as constants
import gman_flags as flags


class LearningRate(object):
    """
    lr
    """

    def __init__(self, initial_learning_rate, save_path):
        """

        Args:
            initial_learning_rate:
            save_path:
        """
        self.path = save_path
        if os.path.exists(save_path):
            self.learning_rate = self.load()
        else:
            self.learning_rate = initial_learning_rate
        self.decay_factor = constants.LEARNING_RATE_DECAY_FACTOR

    def save(self, current_learning_rate):
        """

        Args:
            current_learning_rate:

        Returns:

        """
        current_learning_rate = str(current_learning_rate)
        if not os.path.exists(self.path):
            print("Create Json file for learning rate.")
        learning_rate_file = open(self.path, "w")
        try:
            learning_rate = {'learning_rate': current_learning_rate}
            json.dump(learning_rate, learning_rate_file)
        except IOError as err:
            raise RuntimeError("[Error]: Error happens when read/write " + self.path + ".")
        finally:
            learning_rate_file.close()
        return float(learning_rate["learning_rate"])

    def load(self):
        """

        Returns:

        """
        if not os.path.exists(self.path):
            return constants.INITIAL_LEARNING_RATE
        else:
            # File exist, we need to load the json object
            learning_rate_file = open(self.path, "r")
            try:
                learning_rate_map = json.load(learning_rate_file)
                learning_rate = learning_rate_map["learning_rate"]
            except IOError as err:
                raise RuntimeError("[Error]: Error happens when read/write " + flags.FLAGS.train_json_path + ".")
            finally:
                learning_rate_file.close()
            return float(learning_rate)

    def update(self, current_learning_rate):
        """

        Args:
            current_learning_rate:

        Returns:

        """
        self.save(current_learning_rate)


if __name__ == '__main__':
    pass
