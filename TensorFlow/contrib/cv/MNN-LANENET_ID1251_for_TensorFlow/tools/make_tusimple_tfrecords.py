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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/8 下午8:29
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : make_tusimple_tfrecords.py
# @IDE: PyCharm
"""
Generate cityscapes tfrecords tools
"""
from data_provider import lanenet_data_feed_pipline
from local_utils.log_util import init_logger

LOG = init_logger.get_logger(log_file_name_prefix='generate_tusimple_tfrecords')


def generate_tfrecords():
    """

    :return:
    """
    producer = lanenet_data_feed_pipline.LaneNetDataProducer()
    producer.generate_tfrecords()

    return


if __name__ == '__main__':
    """
    test
    """
    generate_tfrecords()
