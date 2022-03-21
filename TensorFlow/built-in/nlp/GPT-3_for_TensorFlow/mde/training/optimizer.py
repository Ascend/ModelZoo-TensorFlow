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
import tensorflow as tf
from . import optimizer_adamw, optimizer_smdp_adamw   #optimizer_distribute, optimizer_mixedprecision

class Optimizer: 
    def __init__(self, config):
        self.config = config  

    def get_optimizer(self, learning_rate): 
        if self.config['optimizer'] == 'smdp_adamw':
            opt = optimizer_smdp_adamw.AdamWeightDecayOptimizer_with_smdp(
                learning_rate=learning_rate,
                weight_decay_rate=self.config['weight_decay'],
                beta_1=0.9,
                beta_2=0.999, 
                epsilon=1e-6,
                exclude_from_weight_decay=None
                )

        elif self.config['optimizer'] == 'adamw':
            opt = optimizer_adamw.AdamWeightDecayOptimizer(
                learning_rate=learning_rate,
                weight_decay_rate=self.config['weight_decay'],
                beta_1=0.9,
                beta_2=0.999, 
                epsilon=1e-6,
                exclude_from_weight_decay=None)

        else:
            print ('ERROR: no base Optimizer!!!')
            raise
       
#        opt = optimizer_distribute.DistributeOptimizer(
#                optimizer = opt,
#                iter_size=self.config['iter_size'][0], 
#                accum_dtype=self.config['dtype'])

        # if self.config['dtype'] == tf.float16:                                   #当前在main函数中手动乘loss scale，故此处不需要套混合精度opt
        #     opt = optimizer_mixedprecision.MixedPrecisionOptimizer_version2(
        #         optimizer=opt,
        #         config=self.config)

        return opt
