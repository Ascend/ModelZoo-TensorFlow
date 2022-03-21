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
import numpy as np


def get_loss_from_om_output(om_output_dir):
    # get om output files
    outputs = os.listdir(om_output_dir)
    outputs.sort()
    outputs_data = list()

    for i in range(len(outputs)//4):
        data = 0.0
        for j in range(4):
            output_data = np.fromfile(os.path.join(om_output_dir,outputs[i*4+j]),dtype=np.float32)
            data += output_data
        outputs_data.extend(data)
    outputs_data = np.array(outputs_data)
    return np.mean(outputs_data)


mean_loss = get_loss_from_om_output("/home/test_user06/struct-vrn/om/om_output/20210804_173815")
print("[INFO] Mean losss is {}".format(mean_loss))
