# Copyright 2018 Google Inc. All Rights Reserved.
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
import tensorflow as tf
class _LogSessionRunHook(tf.train.SessionRunHook):
    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs(
                fetches=['overflow_status_reduce_all:0', 'loss_scale:0', 'learning_rate/ToFloat:0', 'learning_rate/Exp:0'])

    def after_run(self, run_context, run_values):
        print('ToFloat=', run_values.results[2], ' Exp=', run_values.results[3], flush=True)
        if not run_values.results[0]:
            print('Find overflow in this step, skip apply gradients, loss scale value=%d' % run_values.results[1], flush=True)
        else:
            print('Apply gradients, loss scale value=%d' % run_values.results[1], flush=True)
