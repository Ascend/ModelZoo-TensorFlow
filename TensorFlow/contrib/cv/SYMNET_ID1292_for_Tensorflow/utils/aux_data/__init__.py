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
import importlib

def load_loss_weight(dataset_name):
    """Loss weight to balance the categories
    weight = -log(frequency)"""

    if dataset_name[-1]=='g':
        dataset_name = dataset_name[:-1]
    
    try:
        Weight = importlib.import_module('utils.aux_data.%s_weight'%dataset_name)
        
        if 'pair_weight' in Weight.__dict__:
            return Weight.attr_weight, Weight.obj_weight, Weight.pair_weight
        else:
            return Weight.attr_weight, Weight.obj_weight, None

    except ImportError:
        raise NotImplementedError("Loss weight for %s is not implemented yet"%dataset_name)


def load_wordvec_dict(dataset_name, vec_type):
    if dataset_name[-1]=='g':
        dataset_name = dataset_name[:-1]
        
    try:
        Wordvec = importlib.import_module('utils.aux_data.%s_%s'%(vec_type, dataset_name))
        return Wordvec.attrs_dict, Wordvec.objs_dict

    except ImportError:
        raise NotImplementedError("%s vector for %s is not ready yet"%(vec_type, dataset_name))
