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

from npu_bridge.npu_init import *
import tensorflow as tf


class MultiDataset(object):
    #  A class to combine multi dataset input
    def __init__(self, db_list):
        assert type(db_list) == list and len(db_list) >= 1
        self.db_list = db_list

    def get(self, name_wanted):
        data_list = []
        for i, db in enumerate(self.db_list):
            data = db.get()
            data_list.append(data)

        ret_data = {}
        for name in name_wanted:
            ret_data[name] = tf.concat([d[name] for d in data_list], axis=0)

        return ret_data


def combineMultiDataset(data_list, name_wanted):
    # data_list is a list of data_dict
    ret_data = {}
    for name in name_wanted:
        ret_data[name] = tf.concat([d[name] for d in data_list], axis=0)

    return ret_data


if __name__ == '__main__':
    pass

