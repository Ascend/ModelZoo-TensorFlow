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
from npu_bridge.npu_init import *
from utils import inject_tfkeras_modules, init_tfkeras_custom_objects
import efficientnet as model

EfficientNetB0 = inject_tfkeras_modules(model.EfficientNetB0)
EfficientNetB1 = inject_tfkeras_modules(model.EfficientNetB1)
EfficientNetB2 = inject_tfkeras_modules(model.EfficientNetB2)
EfficientNetB3 = inject_tfkeras_modules(model.EfficientNetB3)
EfficientNetB4 = inject_tfkeras_modules(model.EfficientNetB4)
EfficientNetB5 = inject_tfkeras_modules(model.EfficientNetB5)
EfficientNetB6 = inject_tfkeras_modules(model.EfficientNetB6)
EfficientNetB7 = inject_tfkeras_modules(model.EfficientNetB7)

preprocess_input = inject_tfkeras_modules(model.preprocess_input)

init_tfkeras_custom_objects()

