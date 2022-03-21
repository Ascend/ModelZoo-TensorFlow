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

# from .Criteo import Criteo
from .iPinYou import iPinYou
# from .Avazu import Avazu
# # from .Huawei import Huawei
# from .Criteo_all import Criteo_all
# from .Criteo_Challenge import Criteo_Challenge


def as_dataset(data_name, initialized=True):
    data_name = data_name.lower()
    # if data_name == 'criteo':
    #     return Criteo(initialized=initialized)
    if data_name == 'ipinyou':
        return iPinYou(initialized=initialized)
    # elif data_name == 'avazu':
    #     return Avazu(initialized=initialized)
    # elif data_name == 'criteo_9d':
    #     return Criteo_all(initialized=initialized, num_of_days=9)
    # elif data_name == 'criteo_16d':
    #     return Criteo_all(initialized=initialized, num_of_days=16)
    # elif data_name == 'criteo_challenge':
    #     return Criteo_Challenge(initialized=initialized)
    # elif data_name == 'huawei':
    #     return Huawei(initialized=initialized)
