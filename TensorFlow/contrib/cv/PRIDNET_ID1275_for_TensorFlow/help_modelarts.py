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
import moxing as mox


def modelarts_result2obs(Flags):
    work_dir = os.getcwd()
    obs_result_dir = os.path.join(Flags.obs_dir, "result")
    obs_checkpoint = os.path.join(Flags.obs_dir, "checkpoint")
    if not mox.file.exists(obs_result_dir):
        mox.file.make_dirs(obs_result_dir)
    if not mox.file.exists(obs_checkpoint):
        mox.file.make_dirs(obs_checkpoint)
    mox.file.copy_parallel(src_url=Flags.result, dst_url=obs_result_dir)
    mox.file.copy_parallel(src_url=Flags.train_model_url, dst_url=obs_checkpoint)
    # copy profiling data from modelarts to obs
    if Flags.profiling:
        modelarts_profiling_dir = os.path.join(work_dir, "npu_profiling")
        obs_profiling_dir = os.path.join(Flags.obs_dir, "npu_profiling")
        if not mox.file.exists(obs_profiling_dir):
            mox.file.make_dirs(obs_profiling_dir)
        mox.file.copy_parallel(src_url=modelarts_profiling_dir, dst_url=obs_profiling_dir)

