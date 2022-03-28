# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
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
from datetime import datetime
from collections import defaultdict
import threading
import time
import logging
import os

#from pandas import DataFrame
from collections import defaultdict


class TrainLog:
    """Saves training logs in Pandas msgpacks"""

    INCREMENTAL_UPDATE_TIME = 300

    def __init__(self, directory, name):
        self.log_file_path = "{}/{}.msgpack".format(directory, name)
        self._log = defaultdict(dict)
        self._log_lock = threading.RLock()
        self._last_update_time = time.time() - self.INCREMENTAL_UPDATE_TIME

    def record_single(self, step, column, value):
        self._record(step, {column: value})

    def record(self, step, col_val_dict):
        self._record(step, col_val_dict)

    #def save(self):
        #df = self._as_dataframe()
        #df.to_msgpack(self.log_file_path, compress='zlib')

    def _record(self, step, col_val_dict):
        with self._log_lock:
            self._log[step].update(col_val_dict)
            if time.time() - self._last_update_time >= self.INCREMENTAL_UPDATE_TIME:
                self._last_update_time = time.time()
                #self.save()

    # def _as_dataframe(self):
    #     with self._log_lock:
    #         return DataFrame.from_dict(self._log, orient='index')


class RunContext:
    """Creates directories and files for the run"""

    def __init__(self, runner_file, run_idx, result_pah):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        # runner_name = os.path.basename(runner_file).split(".")[0]
        # self.result_dir = "{root}/{runner_name}/{date:%Y-%m-%d_%H:%M:%S}/{run_idx}".format(
        #     root='results',
        #     runner_name=runner_name,
        #     date=datetime.now(),
        #     run_idx=run_idx
        # )
        self.result_dir = result_pah
        self.transient_dir = self.result_dir + "/transient"
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        os.makedirs(self.transient_dir)

    def create_train_log(self, name):
        return TrainLog(self.result_dir, name)

