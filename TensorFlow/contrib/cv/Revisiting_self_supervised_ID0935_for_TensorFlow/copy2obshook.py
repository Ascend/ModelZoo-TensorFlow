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

from __future__ import print_function
import tensorflow as tf
import logging
import numpy as np
import time
import sys,os
import absl.app as app
import absl.flags as flags
import absl.logging as logging
#import moxing as mox

class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""
    FLAGS = flags.FLAGS
    def __init__(self,FLAGS):
        self.FLAGS = FLAGS


    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self,run_context):
        self._step += 1


    def after_run(self,run_context,run_values):
        if((self._step%1000)==0):
            print("复制checkpiont到obs==================>")
            modelarts_result2obs(self.FLAGS)


def modelarts_result2obs(FLAGS):
    """
    Copy debug data from modelarts to obs.
    According to the swich flags, the debug data may contains auto tune repository,
    dump data for precision comparision, even the computation graph and profiling data.
    """
    work_dir = os.getcwd()
    print("workdir==",work_dir)
    ## copy result from modelarts to obs
    obs_result_dir = os.path.join(FLAGS.obs_url, 'result')
    #if not mox.file.exists(obs_result_dir):
        #mox.file.make_dirs(obs_result_dir)
    print(FLAGS.workdir)
    print(obs_result_dir)
    #mox.file.copy_parallel(src_url=FLAGS.workdir, dst_url=obs_result_dir)
    #print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(FLAGS.workdir, obs_result_dir))

    ## Copy profiling data. Comment this snippets if npu_profiling is off.
    """if FLAGS.profiling:
        modelarts_profiling_dir = os.path.join(work_dir, "npu_profiling")
        obs_profiling_dir = os.path.join(FLAGS.obs_dir, 'npu_profiling')
        if not mox.file.exists(obs_profiling_dir):
            mox.file.make_dirs(obs_profiling_dir)
        mox.file.copy_parallel(modelarts_profiling_dir, obs_profiling_dir)
        print("===>>>Profiling data:{} on OBS dir:{}".format(mox.file.list_directory(obs_profiling_dir), obs_profiling_dir))
        """