"""
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
"""
import logging
import logging.config
import os.path
from collections import OrderedDict

import tensorflow as tf
import yaml

from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def config_logging(logfile=None):
    path = os.path.join(os.path.dirname(__file__), 'logging.yml')
    with open(path, 'r') as f:
        config = yaml.load(f.read())
    if logfile is None:
        del config['handlers']['file_handler']
        del config['root']['handlers'][-1]
    else:
        config['handlers']['file_handler']['filename'] = logfile
    logging.config.dictConfig(config)


def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])

def collect_vars(scope, start=None, end=None, prepend_scope=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(end):
                end = i
                break
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        var_dict[var_name] = var
    return var_dict
