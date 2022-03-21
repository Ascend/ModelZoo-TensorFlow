#
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
import importlib

from dataset.util.tfrecord import build_tfrecords

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build tfrecords')
    parser.add_argument('config')
    config = importlib.import_module(parser.parse_args().config)
    example_parser = getattr(importlib.import_module(config.PARSER[0]), config.PARSER[1])
    for name, kwargs in config.BUILD_TFRECORDS.items():
        print(build_tfrecords(config.TFRECORD_PATH / name, name, kwargs['num_shards'], example_parser(**kwargs)))
