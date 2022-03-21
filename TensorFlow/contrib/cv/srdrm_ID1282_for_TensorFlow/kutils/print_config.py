#!/usr/bin/python3
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
# -*- coding: utf-8 -*-
# @Time    : 2021/9/13 10:20
# @Author  : XJTU-zzf
# @FileName: print_config.py

def print_config_train(Flags, log):
    log.info("===================Train task Config==================")
    log.info("| Model: {}".format(Flags.model_name))
    log.info("| Train mode: {}".format(Flags.train_mode))
    log.info("| Platform: {}".format(Flags.platform))
    log.info("| chip: {}".format(Flags.chip))
    log.info("| data_path: {}".format(Flags.data_path))
    log.info("| num_epochs: {}".format(Flags.num_epochs))
    log.info("| batch_size: {}".format(Flags.batch_size))
    log.info("| sample_interval: {}".format(Flags.sample_interval))
    log.info("| ckpt_interval: {}".format(Flags.ckpt_interval))

    if Flags.chip == 'npu':
        log.info("| profiling: {}".format(Flags.profiling))
        log.info("| obs_dir: {}".format(Flags.obs_dir))
        log.info("| Modelarts result path: {}".format(Flags.result))
    log.info("======================================================")


def print_config_test(Flags,log):
    log.info("===================Test task Config==================")
    log.info("| Model: {}".format(Flags.model_name))
    log.info("| test mode: {}".format(Flags.test_mode))
    log.info("| Platform: {}".format(Flags.platform))
    log.info("| chip: {}".format(Flags.chip))
    log.info("| test_data_path: {}".format(Flags.data_dir))
    log.info("| test_epoch: {}".format(Flags.test_epoch))

    if Flags.chip == 'npu':
        log.info("| result_dir: {}".format(Flags.result))
        log.info("| obs_dir: {}".format(Flags.obs_dir))
    log.info("=====================================================")
